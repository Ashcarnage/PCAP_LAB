#include<stdio.h>
#include<stdlib.h>
#include"mpi.h"

int factorial(int n){
    int val = 1;
    for (int i=1;i<=n;i++){
        val*=i;
    }
    return val;
}
void custom_error_handler(MPI_Comm *comm, int *err_code,...){
    char error_string[MPI_MAX_ERROR_STRING];
    int len_of_error;
    MPI_Error_string(*err_code, error_string, &len_of_error);
    fprintf(stderr, "Custom Error Handler: %s\n", error_string);
    MPI_Abort(*comm, *err_code);
}

int main(int argc, char *argv[]){
    int size, rank, n,fact, scan_result;
    int error_code,error_class;
    MPI_Init(&argc,&argv);
    MPI_Comm_size( MPI_COMM_WORLD ,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    // MPI_Error_class(error_code,&error_class); 
    MPI_Errhandler errhandler;
    MPI_Datatype MPI_int;
    // MPI_Comm_create_errhandler(custom_error_handler, &errhandler);
    // MPI_Comm_set_errhandler(MPI_COMM_WORLD, errhandler);
    MPI_Errhandler_set(MPI_COMM_WORLD,MPI_ERRORS_RETURN);
    int local_factorial = factorial(rank+1);
    int err = MPI_Scan(&local_factorial,&scan_result,1,MPI_int,MPI_SUM,MPI_COMM_WORLD);
    if(err != MPI_SUCCESS){
        char error_str[50];
        int length_of_error_str;
        MPI_Error_class(err,&error_class);
        MPI_Error_string(error_class,error_str, &length_of_error_str);
        printf("helloooo Rank %d: Error during MPI_Scan: %s\n", rank, error_str);
        MPI_Abort(MPI_COMM_WORLD, err);
        return EXIT_FAILURE;
    }
    printf("Process %d: Local factorial = %d | Prefix sum = %d\n",rank,local_factorial,scan_result);
    if(rank==size-1){
        printf("The final sum : %d",scan_result);
    }
    MPI_Finalize();
    return 0;
}