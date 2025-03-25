#include<stdio.h>
#include<stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void modifyMatrix(float *A, int M, int N){
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    if(row<M){
        int row_offest = row*N;
        for(int col=0;col<N;col++){
            A[row_offest+col] = powf(A[row_offest+col],row+1);
        }
    }
}

void ProcessMatrixCuda(float *h_A,int M, int N){
    float *d_A;
    int size  = M*N*sizeof(float);

    cudaMalloc((void**)&d_A,size);
    cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);

    int blockSize = 16;
    int gridSize = (M + blockSize - 1) / blockSize;

    modifyMatrix<<<gridSize,blockSize>>>(d_A,M,N);

    cudaMemcpy(h_A,d_A,size,cudaMemcpyDeviceToHost);
    cudaFree(d_A);  
}

int main(){
    int M = 4;
    int N = 5;
    float h_A[4][5] = {
        {1, 2, 3, 4, 5},
        {6, 7, 8, 9, 10},
        {11, 12, 13, 14, 15},
        {16, 17, 18, 19, 20}
    };
    printf("Original Matrix:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", h_A[i][j]);
        }
        printf("\n");
    }

    ProcessMatrixCuda((float*)h_A,M,N);

    printf("\nModified Matrix:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", h_A[i][j]);
        }
        printf("\n");
    }

    return 0;

    

}