// #include<stdio.h>
// #include"mpi.h"
// #include<stdlib.h>

// int main (int argc, char *argv []){
//     int rank,M,size, n,c;
//     int *arr = NULL;
//     int *subdata = NULL;
//     float *averages = NULL;
//     float total_avg = 0.0;
//     MPI_Init(&argc, &argv);
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     if (rank == 0){
//         printf("Enter an Integer value for M : ");
//         scanf("%d",&M);
//         arr = (int*)malloc(size*M*sizeof(int));
//         n = size;
//         printf("Enter %d x %d elements : \n",n,M);\
//         for(int i = 0;i<n*M;i++){
//             scanf("%d",&arr[i]);
//         }

//     }
//     subdata = (int*)malloc(sizeof(int)*M);

//     MPI_Scatter(arr,M,MPI_INT,subdata,M,MPI_INT,0,MPI_COMM_WORLD);

//     float sum = 0.0;
//     for (int i =0; i<M;i++){
//         sum+=subdata[i];
//     }
//     float avg = sum/M;
//     if (rank == 0){
//         averages = (float*)malloc(sizeof(float)*size);
//     }
//     MPI_Gather(&avg, 1, MPI_FLOAT, averages, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
//     if (rank == 0){
//         float total_sum = 0.0;
//         for (int i = 0; i<size;i++){
//             total_sum+=averages[i];
//         }
//         total_avg = total_sum/size;
//         printf("The Total average of all M averages is : %f\n",total_avg);
//             free(arr);
//             free(averages);
//     }
//     free(subdata);;
//     MPI_Finalize();
//     return 0;
// }


#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int M, N; // M: number of elements, N: number of processes
    int *data = NULL;  // Pointer to hold the N x M array on root process
    int *sub_data = NULL; // Array to hold M elements per process
    float *averages = NULL; // Array to hold averages computed by each process
    float *final_averages = NULL; // To hold the gathered averages at root
    float total_average = 0.0;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Root process: Input the values of M and prepare the data
        printf("Enter the value of M (number of elements per process): ");
        scanf("%d", &M);
        N = size; // N is the number of processes
        
        // Allocate memory for the N x M data array
        data = (int *)malloc(M * N * sizeof(int));
        
        printf("Enter the %d x %d elements:\n", N, M);
        for (int i = 0; i < N * M; i++) {
            scanf("%d", &data[i]);
        }
    }

    // Allocate space for each process to store M elements
    sub_data = (int *)malloc(M * sizeof(int));

    // Scatter the data array to all processes
    MPI_Scatter(data, M, MPI_INT, sub_data, M, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process computes the average of its M elements
    float local_sum = 0.0;
    for (int i = 0; i < M; i++) {
        local_sum += sub_data[i];
    }
    float local_average = local_sum / M;

    // Allocate memory on root process to collect the averages
    if (rank == 0) {
        averages = (float *)malloc(size * sizeof(float));
    }

    // Gather all local averages to root process
    MPI_Gather(&local_average, 1, MPI_FLOAT, averages, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Root process computes the total average of all averages
        float sum_of_averages = 0.0;
        for (int i = 0; i < size; i++) {
            sum_of_averages += averages[i];
        }
        total_average = sum_of_averages / size;

        // Output the total average
        printf("The total average of all the averages is: %f\n", total_average);

        // Clean up memory
        free(data);
        free(averages);
    }

    // Clean up memory for each process
    free(sub_data);

    // Finalize MPI
    MPI_Finalize();
    
    return 0;
}
