#include <stdio.h>
#include <mpi.h>

#define SIZE 4

int main(int argc, char *argv[]) {
    int rank, size;
    int matrix[SIZE][SIZE];
    int local_result[SIZE][SIZE]; // Buffer for storing local results
    int result[SIZE][SIZE];       // Final result to be gathered in root
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Root process enters the matrix
        printf("Process %d: Enter a 4x4 matrix (16 elements):\n", rank);
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                scanf("%d", &matrix[i][j]);
            }
        }
    }

    // Broadcast the matrix to all processes
    MPI_Bcast(&matrix, SIZE * SIZE, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Process %d: Received matrix\n", rank);

    // Perform the transformation: each element is multiplied by its row + column indices
    for (int i = rank; i < SIZE; i += size) {
        for (int j = 0; j < SIZE; j++) {
            local_result[i][j] = matrix[i][j] * (i + j);
            printf("Process %d: Transformed element at (%d, %d) to %d\n", rank, i, j, local_result[i][j]);
        }
    }

    // Gather the result back to the root process
    MPI_Gather(local_result[rank], SIZE * SIZE / size, MPI_INT, result, SIZE * SIZE / size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Process %d: Transformed matrix:\n", rank);
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                printf("%d ", result[i][j]);
            }
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
}
