#include<stdio.h>
#include"mpi.h"
#include<stdlib.h>

int factorial(int n){
    int r = 1;
    for (int i=1;i<=n;i++){
        r=i*r;
    } 
    return r;
}

int main(int argc, char *argv[]){
    int rank, size, n, A[10],c,sum=0;
    int *fact = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank==0){
        n = size;
        printf("Enter a number for %d processes : ",n);
        for(int i=0; i<n;i++){
            scanf("%d",&A[i]);
        }
    }
    MPI_Scatter(A, 1, MPI_INT, &c, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int factNum = factorial(c);
    if (rank == 0){
        fact = (int*)malloc(size*sizeof(int));
    }
    MPI_Gather(&factNum,1,MPI_INT,fact,1,MPI_INT,0,MPI_COMM_WORLD);

    if (rank == 0){
        for(int i=0;i<size;i++){
            sum+=fact[i];
        }
        printf("The sum of factorials is : %d",sum);
        free(fact);
    }
    MPI_Finalize();
    return 0;
}