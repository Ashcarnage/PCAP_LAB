#include<stdio.h>
#include<stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MAX_STR_LEN 100  // Maximum output string length

// CUDA Kernel to generate the output string
__global__ void generateString(char *A, int *B, char *output, int *offsets, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        int idx = row * N + col;
        int startIdx = offsets[idx];  // Starting position in output string

        // Repeat character A[idx] B[idx] times
        for (int i = 0; i < B[idx]; i++) {
            output[startIdx + i] = A[idx];
        }
    }
}

// Host function to compute offsets and process the matrices
void processMatricesCUDA(char *h_A, int *h_B, char *h_output, int M, int N) {
    char *d_A, *d_output;
    int *d_B, *d_offsets;
    int size_A = M * N * sizeof(char);
    int size_B = M * N * sizeof(int);
    int size_output = MAX_STR_LEN * sizeof(char);
    int size_offsets = M * N * sizeof(int);

    // Compute offsets for each character in output string
    int offsets[M * N];
    int totalLength = 0;
    for (int i = 0; i < M * N; i++) {
        offsets[i] = totalLength;
        totalLength += h_B[i];
    }

    // Allocate memory on the GPU
    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMalloc((void **)&d_output, size_output);
    cudaMalloc((void **)&d_offsets, size_offsets);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets, size_offsets, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, size_output);  // Initialize output buffer

    // Define grid and block sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((M + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    // Launch Kernel
    generateString<<<gridSize, blockSize>>>(d_A, d_B, d_output, d_offsets, M, N);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, size_output, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_output);
    cudaFree(d_offsets);
}

// Main function
int main() {
    int M = 2, N = 4;  // Matrix dimensions
    char h_A[M][N] = {
        {'p', 'C', 'a', 'P'},
        {'e', 'X', 'a', 'M'}
    };

    int h_B[M][N] = {
        {1, 2, 4, 3},
        {2, 4, 3, 2}
    };

    char h_output[MAX_STR_LEN] = {0};  // Output string buffer

    printf("Original Matrices:\nA:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%c ", h_A[i][j]);
        }
        printf("\n");
    }

    printf("\nB:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", h_B[i][j]);
        }
        printf("\n");
    }

    // Process matrices on GPU
    processMatricesCUDA(&h_A[0][0], &h_B[0][0], h_output, M, N);

    printf("\nOutput String STR: %s\n", h_output);

    return 0;
}
