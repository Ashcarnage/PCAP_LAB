#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void conv1D(int *N, int *M, float *P, int mask_width, int out_width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < out_width) {
        float Pvalue = 0;
        for (int j = 0; j < mask_width; j++) {
            Pvalue += N[i + j] * M[j];
        }
        P[i] = Pvalue;
    }
}

int main() {
    int N[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
    int M[5] = {1, 2, 3, 4, 5};
    float *P;
    int *d_N, *d_M;
    float *d_P;
    int mask_width = 5;
    int width = 10;
    int out_width = width - mask_width + 1; 

    int size_N = width * sizeof(int);
    int size_M = mask_width * sizeof(int);
    int size_P = out_width * sizeof(float);

    P = (float*)malloc(size_P);


    cudaMalloc((void**)&d_N, size_N);
    cudaMalloc((void**)&d_M, size_M);
    cudaMalloc((void**)&d_P, size_P);

    cudaMemcpy(d_N, N, size_N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, M, size_M, cudaMemcpyHostToDevice);


    int threadsPerBlock = 256;
    int blocksPerGrid = (out_width + threadsPerBlock - 1) / threadsPerBlock;

    conv1D<<<blocksPerGrid, threadsPerBlock>>>(d_N, d_M, d_P, mask_width, out_width);

    cudaMemcpy(P, d_P, size_P, cudaMemcpyDeviceToHost);


    printf("Convolution Output:\n");
    for (int i = 0; i < out_width; i++) {
        printf("%.1f ", P[i]);
    }
    printf("\n");
    cudaFree(d_N);
    cudaFree(d_M);
    cudaFree(d_P);
    free(P);

    return 0;
}
