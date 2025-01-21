#include<stdio.h>
#include"mpi.h"
#include<ctype.h>

int main(int argc, char *argv[]){
    int rank;
    char string[] = "HELLO";
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if (rank<sizeof(string)){
        string[rank] = tolower(string[rank]);
    }
    printf("process %d  : %s\n",rank,string);
    MPI_Finalize();
    return 0;

}