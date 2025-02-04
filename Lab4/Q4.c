#include <stdio.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    char word[100], result[100] = "";
    int N;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Root process enters the word
        printf("Process %d: Enter a word: ", rank);
        scanf("%s", word);
        N = strlen(word);  // Get the length of the word
    }

    // Broadcast the word length to all processes
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Process %d: Received word length %d\n", rank, N);

    // Scatter the characters of the word to all processes
    char local_char;
    MPI_Scatter(word, 1, MPI_CHAR, &local_char, 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    printf("Process %d: Received character '%c'\n", rank, local_char);

    // Each process creates its part of the output
    char local_result[100] = "";
    for (int i = 0; i <= rank; i++) {
        strncat(local_result, &local_char, 1);
    }
    printf("Process %d: Created local result '%s'\n", rank, local_result);

    // Gather the result from all processes
    MPI_Gather(local_result, N, MPI_CHAR, result, N, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Process %d: Transformed word: %s\n", rank, result);
    }

    MPI_Finalize();
    return 0;
}
