#include<stdio.h>
#include<stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// CUDA Kernel for Sparse Matrix-Vector Multiplication (SpMV) using CSR format
__global__ void spmv_csr(int num_rows, const int *row_ptr, const int *col_idx, 
                         const float *values, const float *x, float *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float sum = 0.0f;
        for (int i = row_ptr[row]; i < row_ptr[row + 1]; i++) {
            sum += values[i] * x[col_idx[i]];
        }
        y[row] = sum;
    }
}

// Host function to perform SpMV using CUDA
void spmv_cuda(int num_rows, int nnz, int *h_row_ptr, int *h_col_idx, 
               float *h_values, float *h_x, float *h_y) {
    // Device arrays
    int *d_row_ptr, *d_col_idx;
    float *d_values, *d_x, *d_y;

    // Allocate memory on device
    cudaMalloc((void **)&d_row_ptr, (num_rows + 1) * sizeof(int));
    cudaMalloc((void **)&d_col_idx, nnz * sizeof(int));
    cudaMalloc((void **)&d_values, nnz * sizeof(float));
    cudaMalloc((void **)&d_x, num_rows * sizeof(float));
    cudaMalloc((void **)&d_y, num_rows * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_row_ptr, h_row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, num_rows * sizeof(float), cudaMemcpyHostToDevice);

    // Launch Kernel
    int blockSize = 256;
    int gridSize = (num_rows + blockSize - 1) / blockSize;
    spmv_csr<<<gridSize, blockSize>>>(num_rows, d_row_ptr, d_col_idx, d_values, d_x, d_y);

    // Copy result back to host
    cudaMemcpy(h_y, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
}

int main() {
    // Example: Sparse matrix in CSR format
    int num_rows = 4, nnz = 6;
    int h_row_ptr[] = {0, 2, 4, 5, 6};
    int h_col_idx[] = {0, 1, 1, 2, 3, 2};
    float h_values[] = {10, 20, 30, 40, 50, 60};
    float h_x[] = {1, 2, 3, 4};  // Input vector
    float h_y[4] = {0};  // Output vector

    // Perform SpMV
    spmv_cuda(num_rows, nnz, h_row_ptr, h_col_idx, h_values, h_x, h_y);

    // Print result
    printf("Result: ");
    for (int i = 0; i < num_rows; i++) {
        printf("%f ", h_y[i]);
    }
    printf("\n");

    return 0;
}
