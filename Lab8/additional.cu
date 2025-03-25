#include<stdio.h>
#include<stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define M 3
#define N 2

__global__ void matrixTrans(int *a, int *b){
    int row  = blockIdx.x*blockDim.x + threadIdx.x;
    if (row<N){
        for(int col = 0; col<M; col++){
            if (a[row*M+col]%2==0){
                int total_sum = 0;
                for(int i=0;i<M;i++){
                    total_sum += a[row*M+i];
                }
                b[row*M+col] = total_sum;
            }
            else{
                int total_sum = 0;
                for(int i=0;i<N;i++){
                    total_sum += a[i*M+col];
                }
                b[row*M+col] = total_sum;
            }
        }
    }
}

int main(){
    int a[N][M], b[N][M];
    int *d_a, *d_b;
    int size = N*M*sizeof(int);
    printf("Enter the elements of the matrix:\n");
    for(int i=0;i<N;i++){
        for(int j=0;j<M;j++){
            scanf("%d",&a[i][j]);
        }
    }

    cudaMalloc((void**)&d_a,size);
    cudaMalloc((void**)&d_b,size);

    cudaMemcpy(d_a,a,size,cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N+threadsPerBlock-1)/threadsPerBlock;

    matrixTrans<<<blocksPerGrid,threadsPerBlock>>>(d_a,d_b);

    cudaMemcpy(b,d_b,size,cudaMemcpyDeviceToHost);

    printf("Matrix A:\n");
    for(int i=0;i<N;i++){
        for(int j=0;j<M;j++){
            printf("%d ",a[i][j]);
        }
        printf("\n");
    }

    printf("\nMatrix B:\n");
    for(int i=0;i<N;i++){
        for(int j=0;j<M;j++){
            printf("%d ",b[i][j]);
        }
        printf("\n");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    return 0;
}