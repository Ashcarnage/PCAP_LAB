#include<stdio.h>
#include"mpi.h"
#include<stdlib.h>

int CountOcc(int row[],int val){
    int count = 0;
    for(int i=0;i<3;i++){
        if (val==row[i]){
            count++;
        }
    }
    printf("COUNT : %d",count);
    return count;
}

int main(int argc,char *argv[]){
    int rank,size,n;
    int error_class;
    int total_count;
    MPI_Datatype MPI_int;
    MPI_Init(&argc,&argv);
    MPI_Comm_size( MPI_COMM_WORLD ,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Errhandler_set(MPI_COMM_WORLD,MPI_ERRORS_RETURN);
    int mat[3][3] = {{2,3,3},
                   {4,3,2},
                   {3,3,3}};
    if (rank==0){
        printf("Enter the element to be searched: ");
        scanf("%d",&n);
    }
    MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
    int count = 0;
    for(int i=0;i<3;i++){
        if (n==mat[rank][i]){
            count++;
        }
    }
        // int local_count = CountOcc(mat[rank-1],n);
        // printf("check : %d\n\n\n",local_count);

    int err = MPI_Reduce(&count,&total_count,1,MPI_int,MPI_SUM,0,MPI_COMM_WORLD);
        if(err != MPI_SUCCESS){
            char error_str[50];
            int length_of_error_str;
            MPI_Error_class(err,&error_class);
            MPI_Error_string(error_class,error_str, &length_of_error_str);
            printf("helloooo Rank %d: Error during MPI_Scan: %s\n", rank, error_str);
            MPI_Abort(MPI_COMM_WORLD, err);
            return EXIT_FAILURE;
    } 
    // }
    if (rank==0){
        printf("The element %d has occured %d times in the matrix ",n,total_count);
        
    }
    MPI_Finalize();
    return 0;

}