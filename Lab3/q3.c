#include<stdio.h>
#include"mpi.h"
#include<math.h>

int Add(int a, int b){
    return a+b;
}
int Multiply(int a, int b){
    return a*b;
}
int Power(int a, int b){
    return pow(a,b);
}

int main(int argc, char *argv[]){
    int rank;
    int a=2,b=3;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if (rank==0){
        int n = Add(a,b);
        printf("Adding Numbers %d,%d to get %d\n",a,b,n);
    }
    else if (rank==1){
        int n = Multiply(a,b);
        printf("Multipling Numbers %d,%d to get %d\n",a,b,n);
    }
    else if (rank==2){
        int n = Power(a,b);
        printf("Raising the number %d to %d to get %d\n",a,b,n);
    }
    else printf("No operations performed\n");
    // printf("My rank is %d in total %d process",rank,size);
    MPI_Finalize();
    return 0;

}