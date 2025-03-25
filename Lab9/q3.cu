#include<stdio.h>
#include<stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void onesComplement(int *A, int *B, int M, int N){
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;

    if(row<M && col<N){
        int index = row*N + col;

        if(row==0||row==M-1||col==0||col==N-1){
            B[index] = A[index];
        }
        else{
            int num = A[index];
            int temp = num;
            int bits[32];
            int count = 0;

            while(temp>0){
                bits[count++] = temp%2;
                temp = temp/2;
            }

            int bin=0,place=1;
            for(int i =0;i<count;i++){
                int flipped = bits[i]^1;
                bin +=flipped*place;
                place*=10;
            }
            B[index] = bin;
        }
    }
}

void processMatrix(int *h_A, int *h_B, int M, int N){
    int *d_A, *d_B;
    int size = M*N*sizeof(int);

    cudaMalloc((void**)&d_A,size);
    cudaMalloc((void**)&d_B,size);

    cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);

    dim3 blockSize(16,16);
    dim3 gridSize((M+blockSize.x-1)/blockSize.x,(N+blockSize.y-1)/blockSize.y);

    onesComplement<<<gridSize,blockSize>>>(d_A,d_B,M,N);

    cudaMemcpy(h_B,d_B,size,cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
}

int main(){
    int M=4;
    int N=4;
    int h_A[4][4] = {
        {1, 2, 3, 4},
        {6, 5, 8, 3},
        {2, 4, 10, 1},
        {9, 1, 2, 5}
    };
    int h_B[4][4];
    printf("Original Matrix:\n");
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            printf("%d ",h_A[i][j]);
        }
        printf("\n");
    }
    processMatrix((int*)h_A,(int*)h_B,M,N);

    printf("\nModified Matrix (B):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", h_B[i][j]);
        }
        printf("\n");
    }

    return 0;

}