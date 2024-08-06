#include <algorithm>
#include <fstream>
#include <iostream>
#include <mpi/mpi.h>
#include <random>
#include <string>
#include <vector>

struct Entry {
    int row, col;
    uint64_t value;
};

// Helper function to create an MPI datatype for the Entry struct
MPI_Datatype createEntryType() {
    MPI_Datatype entryType;

    int blocklengths[3] = {1, 1, 1};
    MPI_Aint offsets[3] = {offsetof(Entry, row), offsetof(Entry, col), offsetof(Entry, value)};
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_UINT64_T};

    MPI_Type_create_struct(3, blocklengths, offsets, types, &entryType);
    MPI_Type_commit(&entryType);

    return entryType;
}

// Function to generate sparse matrix
void generateSparseMatrix(std::vector<Entry> &matrix, int n, double sparsity, int seed, int startRow, int rowsPerProc) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::uniform_int_distribution<uint64_t> valueDist(1, 50); // Values between 1 and 50

    for (int i = 0; i < rowsPerProc; ++i) {
        for (int j = 0; j < n; ++j) {
            if (dist(rng) < sparsity) {
                Entry e = {startRow + i, j, valueDist(rng)};
                matrix.push_back(e);
            }
        }
    }
}

// Function to transpose and redistribute the matrix using MPI_Alltoallv
std::vector<Entry> transposeMatrix(std::vector<Entry> &matrix, int n, int size, MPI_Datatype entryType) {
    std::sort(matrix.begin(), matrix.end(),
              [](const Entry &a, const Entry &b) { return a.row < b.row || (a.row == b.row && a.col < b.col); });

    int p = size; // Number of processes
    int rowsPerProc = n / p;
    std::vector<int> sendCounts(p, 0);
    std::vector<int> receiveCounts(p, 0);
    std::vector<int> sendDisplacements(p, 0);
    std::vector<int> receiveDisplacements(p, 0);

    // Calculate send counts
    for (const auto &e: matrix) {
        sendCounts[e.col / rowsPerProc]++;
    }

    // All-to-all to distribute counts
    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, receiveCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Calculate displacements
    for (int i = 1; i < p; i++) {
        sendDisplacements[i] = sendDisplacements[i - 1] + sendCounts[i - 1];
        receiveDisplacements[i] = receiveDisplacements[i - 1] + receiveCounts[i - 1];
    }

    // Prepare buffers for all-to-allv
    std::vector<Entry> sendbuffer(matrix.size());
    std::vector<Entry> recvbuffer(std::accumulate(receiveCounts.begin(), receiveCounts.end(), 0));

    // Organize data to send
    std::vector<int> tempSendDisplacements = sendDisplacements;
    for (const auto &e: matrix) {
        int targetProc = e.col / rowsPerProc;
        sendbuffer[tempSendDisplacements[targetProc]++] = {e.col, e.row, e.value}; // Transpose row/col
    }

    // All-to-allv to perform the transpose
    MPI_Alltoallv(sendbuffer.data(), sendCounts.data(), sendDisplacements.data(), entryType, recvbuffer.data(),
                  receiveCounts.data(), receiveDisplacements.data(), entryType, MPI_COMM_WORLD);

    return recvbuffer;
}

std::vector<uint64_t> multiplyMatrix(const std::vector<Entry> &localA, std::vector<Entry> &localB, int n, int p,
                                     MPI_Comm ringComm, MPI_Datatype entryType) {
    int dst, src;
    MPI_Cart_shift(ringComm, 0, 1, &src, &dst);

    int rowsPerProc = n / p;
    std::vector<uint64_t> localC(rowsPerProc * n, 0);

    std::vector<int> receiveSizes(p, 0);
    int localBSize = localB.size();

    MPI_Allgather(&localBSize, 1, MPI_INT, receiveSizes.data(), 1, MPI_INT, ringComm);

    //    if (rank == 0) {
    //        for (int num : receiveSizes) {
    //            std::cout << num << " ";
    //        }
    //        std::cout << std::endl;
    //    }

    std::vector<Entry> receiveBuffer;

    for (int step = 0; step < p; step++) {
        for (const auto &a: localA) {
            for (const auto &b: localB) {
                if (a.col == b.col) {
                    localC[(a.row % rowsPerProc) * n + b.row] += a.value * b.value;
                }
            }
        }

        receiveBuffer.resize(receiveSizes[(src - step + p) % p]);

        // MPI_Sendrecv_replace(localB.data(), localB.size(), entryType, dst, 0, src, 0, ringComm, MPI_STATUS_IGNORE);

        MPI_Sendrecv(localB.data(), localB.size(), entryType, dst, 0, receiveBuffer.data(), receiveBuffer.size(),
                     entryType, src, 0, ringComm, MPI_STATUS_IGNORE);
        localB.swap(receiveBuffer);
        //
        //        if (rank == 0) { // ensure all rotations complete
        //            MPI_Barrier(ringComm);
        //        }
    }

    return localC;
}

// gather matrix from all the processors
std::vector<Entry> gatherMatrix(const std::vector<Entry> &matrix, int p, MPI_Datatype entryType) {
    std::vector<int> receiveCounts(p, 0);
    int myValue = matrix.size();
    MPI_Allgather(&myValue, 1, MPI_INT, receiveCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> receiveDisplacements(p, 0);
    for (int i = 1; i < p; i++) {
        receiveDisplacements[i] = receiveDisplacements[i - 1] + receiveCounts[i - 1];
    }

    std::vector<Entry> gathered(std::accumulate(receiveCounts.begin(), receiveCounts.end(), 0));

    MPI_Gatherv(matrix.data(), matrix.size(), entryType, gathered.data(), receiveCounts.data(),
                receiveDisplacements.data(), entryType, 0, MPI_COMM_WORLD);

    return gathered;
}

// Convert sparse matrix to dense matrix
void sparseToDense(const std::vector<Entry> &sparseMatrix, std::vector<std::vector<uint64_t>> &denseMatrix) {
    for (const auto &e: sparseMatrix) {
        denseMatrix[e.row][e.col] = e.value;
    }
}

void printMatrix(const std::vector<std::vector<uint64_t>> &denseMatrix, std::ofstream &outFile) {
    for (const auto &row: denseMatrix) {
        for (const auto &val: row) {
            outFile << val << " ";
        }
        outFile << "\n";
    }
    outFile << std::endl;
}

void printOutputVector(const std::vector<uint64_t> &matrix, int n, std::ofstream &outFile) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            outFile << matrix[i * n + j] << " ";
        }
        outFile << "\n";
    }
    outFile << std::endl;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 5) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " n sparsity pf out\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return -1;
    }

    int n = std::stoi(argv[1]);
    double sparsity = std::stod(argv[2]);
    int pf = std::stoi(argv[3]);
    std::string outputFilename = argv[4];

    MPI_Datatype entryType = createEntryType();

    // Generate sparse matrix A and B
    std::vector<Entry> localA;
    std::vector<Entry> localB;
    int rowsPerProc = n / size; // Rows per processor

    generateSparseMatrix(localA, n, sparsity, rank * 100, rank * rowsPerProc, rowsPerProc);
    generateSparseMatrix(localB, n, sparsity, (rank + 1) * 100, rank * rowsPerProc, rowsPerProc);

    double startTime = MPI_Wtime();

    // Transpose matrix B
    std::vector<Entry> transposedB = transposeMatrix(localB, n, size, entryType);

    MPI_Comm ringComm;
    int dims[1] = {size};
    int periods[1] = {1};
    MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, 0, &ringComm);

    // Perform sparse matrix multiplication
    std::vector<uint64_t> localC = multiplyMatrix(localA, transposedB, n, size, ringComm, entryType);

    double endTime = MPI_Wtime();

    std::vector<Entry> gatheredA = gatherMatrix(localA, size, entryType);
    std::vector<Entry> gatheredB = gatherMatrix(localB, size, entryType);
    // std::vector<Entry> gatheredTransB = gatherMatrix(transposedB, n, size, entryType);
    //  Gather the result matrix C
    std::vector<uint64_t> gatheredC(n * n);
    MPI_Gather(localC.data(), rowsPerProc * n, MPI_UINT64_T, gatheredC.data(), rowsPerProc * n, MPI_UINT64_T, 0,
               MPI_COMM_WORLD);

    // Output results
    if (pf == 1) {
        MPI_Barrier(MPI_COMM_WORLD); // Ensure all ranks finish before writing
        if (rank == 0) {
            std::vector<std::vector<uint64_t>> denseMatrixA(n, std::vector<uint64_t>(n, 0));
            std::vector<std::vector<uint64_t>> denseMatrixB(n, std::vector<uint64_t>(n, 0));
            // std::vector<std::vector<uint64_t>> denseMatrixBTrans(n, std::vector<uint64_t>(n, 0));
            sparseToDense(gatheredA, denseMatrixA);
            sparseToDense(gatheredB, denseMatrixB);
            // sparseToDense(gatheredTransB, denseMatrixBTrans);

            std::ofstream outFile(outputFilename);
            printMatrix(denseMatrixA, outFile);
            printMatrix(denseMatrixB, outFile);
            // printMatrix(denseMatrixBTrans, outFile);
            printOutputVector(gatheredC, n, outFile);
            outFile.close();
        }
    }

    if (rank == 0) {
        std::cout << "Time taken: " << (endTime - startTime) * 1000 << "ms" << std::endl;
    }

    MPI_Type_free(&entryType);
    MPI_Comm_free(&ringComm);
    MPI_Finalize();

    return 0;
}
