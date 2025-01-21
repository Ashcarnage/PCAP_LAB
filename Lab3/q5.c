#include <stdio.h>
#include "mpi.h"

int fact(int rank){
    int val = 1;
    for (int i = 1;i<=rank;i++) val*=i;
    return val;
}
void fibbo(int rank){
    int a = 0;
    int b = 1;
    int s = 0;
    printf("%d %d",a,b);
    while(s!=rank && s<=rank){
        s = a+b;
        a = b;
        b = s;
        printf(" %d",s);
    }

}


int main(int argc, char *argv[]){
    int rank;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if(rank%2==0){
        int val = fact(rank);
        printf("The factorial of %d is %d\n",rank,val);
    }
    else{     
        printf("The fibbonaci series of %d is  :",rank);
        fibbo(rank);
    }
    MPI_Finalize();
    return 0;

}
