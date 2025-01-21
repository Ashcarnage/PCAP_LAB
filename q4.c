// #include<stdio.h>
// #include"mpi.h"
// #include<string.h>
// #include<stdlib.h>

// int main(int argc, char *argv[]){
//     int rank, size;
//     char *s1 = NULL;
//     char *s2 = NULL;
//     char *result = NULL;
//     int str_len;

//     MPI_Init(&argc, &argv);
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     if (rank == 0){
//         s1 = (char*)malloc(sizeof(char)*1000);
//         s2 = (char*)malloc(sizeof(char)*1000);
//         printf("Enter string1 (divisible by %d)",size);
//         scanf("%s",s1);
//         printf("Enter string2 (divisible by %d)",size);
//         scanf("%s",s2);
//         int str_len1 = strlen(s1);
//         int str_len2 = strlen(s2);
//         if (str_len1!=str_len2){
//             printf("ERROR: String lengths are not the same");
//         }
//         str_len = str_len1;
//     }
//     MPI_Bcast(&str_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
//     int local_len = str_len/size;
//     char *local_S1 = (char *)malloc(local_len * sizeof(char));
//     char *local_S2 = (char *)malloc(local_len * sizeof(char));
//     char *local_result = (char *)malloc(local_len * sizeof(char));

//     MPI_Scatter(s1, local_len, MPI_CHAR, local_S1, local_len, MPI_CHAR, 0, MPI_COMM_WORLD);
//     MPI_Scatter(s2, local_len, MPI_CHAR, local_S2, local_len, MPI_CHAR, 0, MPI_COMM_WORLD);

//     for(int i=0;i<local_len;i++){
//         local_result[i] = (i%2==0)? local_S1[i]:local_S2[i];
//     }
//     if(rank == 0 ){
//         result = (char*)malloc(sizeof(char)*local_len*2);
//     }
//     MPI_Gather(local_result, local_len*2, MPI_CHAR, result, local_len*2, MPI_CHAR, 0, MPI_COMM_WORLD);
//     if(rank==0){
//         printf("The combined array is : %s",result);
//         free(local_S1);
//         free(local_S2);
//         free(local_result);
//     }
//     free(s1);
//     free(s2);
//     free(result);
//     MPI_Finalize();
//     return 0; 
// }

#include <stdio.h>
#include "mpi.h"
#include <string.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size;
    char *s1 = NULL;
    char *s2 = NULL;
    char *result = NULL;
    int str_len;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Root process: Read strings S1 and S2
        s1 = (char *)malloc(sizeof(char) * 1000);
        s2 = (char *)malloc(sizeof(char) * 1000);
        printf("Enter string1 (divisible by %d): ", size);
        scanf("%s", s1);
        printf("Enter string2 (divisible by %d): ", size);
        scanf("%s", s2);

        int str_len1 = strlen(s1);
        int str_len2 = strlen(s2);
        if (str_len1 != str_len2) {
            printf("ERROR: String lengths are not the same\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        str_len = str_len1;  // Both strings are the same length
    }

    // Broadcast string length to all processes
    MPI_Bcast(&str_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_len = str_len / size;  // Number of characters each process will handle
    printf("This is the local length : %d\n",local_len);
    char *local_S1 = (char *)malloc(local_len * sizeof(char));
    char *local_S2 = (char *)malloc(local_len * sizeof(char));
    char *local_result = (char *)malloc(2*local_len * sizeof(char));

    // Scatter S1 and S2 to all processes
    MPI_Scatter(s1, local_len, MPI_CHAR, local_S1, local_len, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(s2, local_len, MPI_CHAR, local_S2, local_len, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Process the data: combine characters from S1 and S2 alternately
    printf("ahgahahaha  local string 1 : %s done\n",local_S1+'\0');
    for (int i = 0; i < local_len*2-1; i++) {
        local_result[i] = local_S1[i];
        // local_result[i+1] = local_S2[i];
    }
    printf("loacla results: %s\n", local_result);

    if (rank == 0) {
        // Allocate memory for the final result (root process will gather it)
        result = (char *)malloc(str_len * sizeof(char));
    }

    // Gather the combined characters from each process
    MPI_Gather(local_result, local_len, MPI_CHAR, result, str_len*2, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Root process: Print the resultant string
        result[str_len] = '\0';  // Null-terminate the string
        printf("String S1: %s\n", s1);
        printf("String S2: %s\n", s2);
        printf("Resultant String: %s\n"), result;

        // Free allocated memory
        free(s1);
        free(s2);
        free(result);
    }

    // Free local memory (for each process)
    free(local_S1);
    free(local_S2);
    free(local_result);

    MPI_Finalize();
    return 0;
}
