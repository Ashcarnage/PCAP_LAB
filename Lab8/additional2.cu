#include<stdio.h>
#include<stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define N 5  // Matrix size (can be changed)

// Function to calculate factorial
__device__ int factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

// Function to calculate sum of digits
__device__ int sumOfDigits(int n) {
    int sum = 0;
    while (n > 0) {
        sum += n % 10;
        n /= 10;
    }
    return sum;
}

// Kernel to transform the matrix
__global__ void transformMatrix(int *matrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < N && idy < N) {
        int index = idy * N + idx;
        int value = matrix[index];
        
        if (idx == idy) {
            // Principal diagonal - replace with zero
            matrix[index] = 0;
        } else if (idx > idy) {
            // Above principal diagonal - replace with factorial
            matrix[index] = factorial(value);
        } else {
            // Below principal diagonal - replace with sum of digits
            matrix[index] = sumOfDigits(value);
        }
    }
}

int main() {
    int h_matrix[N][N];
    int *d_matrix;
    
    // Initialize the matrix with some values
    printf("Original Matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_matrix[i][j] = i + j + 1;  // Simple initialization
            printf("%d\t", h_matrix[i][j]);
        }
        printf("\n");
    }
    
    // Allocate memory on the device
    cudaMalloc((void**)&d_matrix, N * N * sizeof(int));
    
    // Copy the matrix from host to device
    cudaMemcpy(d_matrix, h_matrix, N * N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    
    // Launch the kernel
    transformMatrix<<<gridDim, blockDim>>>(d_matrix);
    
    // Copy the result back to host
    cudaMemcpy(h_matrix, d_matrix, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print the transformed matrix
    printf("\nTransformed Matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d\t", h_matrix[i][j]);
        }
        printf("\n");
    }
    
    // Free device memory
    cudaFree(d_matrix);
    
    return 0;
}
