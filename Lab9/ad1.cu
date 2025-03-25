#include<stdio.h>
#include<stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define M 2  // Number of rows
#define N 3  // Number of columns

// CUDA Kernel to compute B[i][j] = sum of elements in row i + sum of elements in column j
__global__ void computeB(int *A, int *B) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        int rowSum = 0, colSum = 0;

        // Compute sum of row elements
        for (int j = 0; j < N; j++)
            rowSum += A[row * N + j];

        // Compute sum of column elements
        for (int i = 0; i < M; i++)
            colSum += A[i * N + col];

        // Store in output matrix
        B[row * N + col] = rowSum + colSum - A[row * N + col];
    }
}

// Host function to process the matrix
void processMatrixCUDA(int *h_A, int *h_B) {
    int *d_A, *d_B;
    int size = M * N * sizeof(int);

    // Allocate memory on the GPU
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);

    // Copy matrix A to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    // Define grid and block sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((M + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    // Launch Kernel
    computeB<<<gridSize, blockSize>>>(d_A, d_B);

    // Copy result back to host
    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
}

// Main function
int main() {
    int h_A[M][N] = {
        {1, 2, 3},
        {4, 5, 6}
    };

    int h_B[M][N];  // Output matrix

    printf("Original Matrix (A):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", h_A[i][j]);
        }
        printf("\n");
    }

    // Process matrix on GPU
    processMatrixCUDA(&h_A[0][0], &h_B[0][0]);

    printf("\nOutput Matrix (B):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", h_B[i][j]);
        }
        printf("\n");
    }

    return 0;
}