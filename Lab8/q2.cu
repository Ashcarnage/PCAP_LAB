#include<stdio.h>
#include<stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Matrix dimensions
#define N 4  
#define M 3  
#define K 2  
// Kernel for row-wise multiplication 
__global__ void matrixMultiplyByRow(int *a, int *b, int *c, int widthA, int widthB) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N) {
        for (int col = 0; col < widthB; col++) {
            int sum = 0;
            for (int k = 0; k < widthA; k++) {
                sum += a[row * widthA + k] * b[k * widthB + col];
            }
            c[row * widthB + col] = sum;
        }
    }
}

// Kernel for column-wise multiplication (each thread computes one column)
__global__ void matrixMultiplyByColumn(int *a, int *b, int *c, int heightA, int widthA, int widthB) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < widthB) {
        for (int row = 0; row < heightA; row++) {
            int sum = 0;
            for (int k = 0; k < widthA; k++) {
                sum += a[row * widthA + k] * b[k * widthB + col];
            }
            c[row * widthB + col] = sum;
        }
    }
}

// Kernel for element-wise multiplication (each thread computes one element)
__global__ void matrixMultiplyByElement(int *a, int *b, int *c, int widthA, int widthB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < widthB) {
        int sum = 0;
        for (int k = 0; k < widthA; k++) {
            sum += a[row * widthA + k] * b[k * widthB + col];
        }
        c[row * widthB + col] = sum;
    }
}

int main() {
    int a[N][K], b[K][M], c[N][M];
    int *d_a, *d_b, *d_c;
    
    // Allocate memory on device
    cudaMalloc((void**)&d_a, N * K * sizeof(int));
    cudaMalloc((void**)&d_b, K * M * sizeof(int));
    cudaMalloc((void**)&d_c, N * M * sizeof(int));
    
    // Initialize matrices on host
    printf("Matrix A (%dx%d):\n", N, K);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            a[i][j] = i + j + 1;
            printf("%d\t", a[i][j]);
        }
        printf("\n");
    }
    
    printf("\nMatrix B (%dx%d):\n", K, M);
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < M; j++) {
            b[i][j] = i + j + 1;
            printf("%d\t", b[i][j]);
        }
        printf("\n");
    }
    
    // Copy data from host to device
    cudaMemcpy(d_a, a, N * K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, K * M * sizeof(int), cudaMemcpyHostToDevice);
    
    // a. Each row computed by one thread
    printf("\na. Matrix multiplication by row:\n");
    dim3 blockDim1D(256);
    dim3 gridDim1D((N + blockDim1D.x - 1) / blockDim1D.x);
    matrixMultiplyByRow<<<gridDim1D, blockDim1D>>>(d_a, d_b, d_c, K, M);
    cudaMemcpy(c, d_c, N * M * sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            printf("%d\t", c[i][j]);
        }
        printf("\n");
    }
    
    // b. Each column computed by one thread
    printf("\nb. Matrix multiplication by column:\n");
    dim3 gridDim1D_col((M + blockDim1D.x - 1) / blockDim1D.x);
    matrixMultiplyByColumn<<<gridDim1D_col, blockDim1D>>>(d_a, d_b, d_c, N, K, M);
    cudaMemcpy(c, d_c, N * M * sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            printf("%d\t", c[i][j]);
        }
        printf("\n");
    }
    
    // c. Each element computed by one thread
    printf("\nc. Matrix multiplication by element:\n");
    dim3 blockDim2D(16, 16);
    dim3 gridDim2D((M + blockDim2D.x - 1) / blockDim2D.x, 
                   (N + blockDim2D.y - 1) / blockDim2D.y);
    matrixMultiplyByElement<<<gridDim2D, blockDim2D>>>(d_a, d_b, d_c, K, M);
    cudaMemcpy(c, d_c, N * M * sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            printf("%d\t", c[i][j]);
        }
        printf("\n");
    }
    
    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}
