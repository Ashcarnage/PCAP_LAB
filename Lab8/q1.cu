#include<stdio.h>
#include<stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 5
#define M 5
//Row-wise matrix addition
__global__ void matrixAddByRow(int *a, int*b, int *c, int width){
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    if(row<width){
        for(int col = 0; col<width; col++){
            c[row*width+col] = a[row*width+col] + b[row*width+col];
        }
    }
}
//Column-wise matrix addition

__global__ void matrixAddByCol(int *a, int*b, int *c, int width){
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if(col<width){
        for(int row = 0; row<width; row++){
            c[row*width+col] = a[row*width+col] + b[row*width+col];
        }
    }
}

// Element-wise matrix addition

__global__ void matrixAddByElement(int *a, int*b, int *c, int n, int m){
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;
    if(row<n && col<m){
        c[row*m+col] = a[row*m+col] + b[row*m+col];
    }
}

int main(){
    int a[N][M], b[N][M], c[N][M];
    int *d_a, *d_b, *d_c;
    int size = N*M*sizeof(int);

    for(int i=0;i<N;i++){
        for(int j=0;j<M;j++){
            a[i][j] = i+j;
            b[i][j] = i*j;
        }
    }
    printf("Matrix A:\n");
    for(int i=0;i<N;i++){
        for(int j=0;j<M;j++){
            printf("%d ",b[i][j]);
        }
        printf("\n");
    }
    printf("\nMatrix B:\n");
    for(int i=0;i<N;i++){
        for(int j=0;j<M;j++){
            printf("%d ",c[i][j]);
        }
        printf("\n");
    }

    cudaMalloc((void**)&d_a,size);
    cudaMalloc((void**)&d_b,size);
    cudaMalloc((void**)&d_c,size);

    cudaMemcpy(d_a,a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b,size,cudaMemcpyHostToDevice);

    printf("Matrix Addition by Row:\n");
    dim3 blockDim1D(256);
    dim3 gridDim1D((N+blockDim1D.x-1)/blockDim1D.x);
    matrixAddByRow<<<gridDim1D,blockDim1D>>>(d_a,d_b,d_c,N);
    cudaMemcpy(c,d_c,size,cudaMemcpyDeviceToHost);
    for(int i=0;i<N;i++){
        for(int j=0;j<M;j++){
            printf("%d ",c[i][j]);
        }
        printf("\n");
    }

    printf("\nMatrix Addition by Column:\n");
    matrixAddByCol<<<gridDim1D,blockDim1D>>>(d_a,d_b,d_c,M);
    cudaMemcpy(c,d_c,size,cudaMemcpyDeviceToHost);
    for(int i=0;i<N;i++){
        for(int j=0;j<M;j++){
            printf("%d ",c[i][j]);
        }
        printf("\n");
    }

    printf("\nMatrix Addition by Element:\n");
    dim3 blockDim2D(16,16);
    dim3 gridDim2D((N+blockDim2D.x-1)/blockDim2D.x,(M+blockDim2D.y-1)/blockDim2D.y);
    matrixAddByElement<<<gridDim2D,blockDim2D>>>(d_a,d_b,d_c,N,M);
    cudaMemcpy(c,d_c,size,cudaMemcpyDeviceToHost);
    for(int i=0;i<N;i++){
        for(int j=0;j<M;j++){
            printf("%d ",c[i][j]);
        }
        printf("\n");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}