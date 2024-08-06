# Georgia Tech CSE 6220 Lab 3: Sparse Square Matrix Multiplication

This project implements the parallel sparse square matrix multiplication using the Message Passing Interface (MPI) library.
It implements an efficient algorithm to multiply large sparse matrices stored in a special format.

## Usage

To compile the program, run the following command:
```shell
mpicxx -std=c++11 -o spmat matmul.cpp
```

To run the program, use the following command:
```shell
mpirun -np <num_processes> ./spmat <matrix_size> <sparsity> <print_flag> <output_file>
```

- `num_processes`: The number of MPI processes to use.
- `matrix_size`: The dimension of the square matrices (n x n).
- `sparsity`: The sparsity parameter ∈ (0, 1] representing the probability of a matrix element being non-zero.
- `print_flag`: A flag (0 or 1) indicating whether to print the input and output matrices to a file.
- `output_file`: The name of the output file to write the matrices to (if print_flag is 1).

Example:

```shell
mpirun -np 4 ./spmat 16 0.1 1 output.txt
```

## How the Program Works

1. The program starts by generating two sparse matrices `A` and `B` using a random number generator based on the provided matrix size and sparsity parameter. Each process generates a portion of the matrices.
2. The sparse matrices are stored using a special format where only the non-zero elements are stored as `(row, col, value)` tuples.
3. The matrix `B` is transposed and redistributed among the processes using `MPI_Alltoallv` to optimize the communication pattern.
4. The matrix multiplication is performed using a ***ring topology*** created with `MPI_Cart_create`. Each process multiplies its local portion of matrix `A` with the corresponding portion of the transposed matrix `B`.
5. The ring topology is used to rotate the transposed matrix `B` among the processes, allowing each process to multiply with the appropriate portion of `B`.
6. The resulting matrix `C` is stored in a sparse format, and the non-zero elements are gathered from all processes using `MPI_Gatherv`.
7. If the print flag is set to `1`, the program converts the sparse matrices `A`, `B`, and `C` to dense format and writes them to the specified output file.
8. The program measures the execution time and prints it on the console.

## Machine Used

The program was mainly developed on the following local machine:

- Operating System: Ubuntu 22.04 Virtual Machine
- Compiler: mpic++ (MPI C++ Compiler)
- MPI Library: OpenMPI

It has also been tested on the [PACE ICE](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042102) computing clusters.
