#include<stdio.h>
#include"mpi.h"
#include<math.h>

int Pow(int x, int rank){
    return pow(x,rank);
}

int main(int argc, char *argv[]){
    int rank;
    int constant = 2;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    int value = Pow(constant,rank);
    printf("The answer of %d raised to %d is %d",constant,rank,value);
    return 0;
}