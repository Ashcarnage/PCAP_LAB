#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Kernel to perform one phase (even or odd) of the odd-even transposition sort.
// phase = 0 -> even phase (compare indices 0 & 1, 2 & 3, ...)
// phase = 1 -> odd phase (compare indices 1 & 2, 3 & 4, ...)
__global__ void oddEvenSortPhase(int *d_arr, int n, int phase) {
    // Calculate a unique thread id.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int index;
    if (phase == 0) {
        // Even phase: starting index is even.
        index = 2 * tid;
    } else {
        // Odd phase: starting index is odd.
        index = 2 * tid + 1;
    }
    
    // If the pair is within the bounds of the array, compare and swap if needed.
    if (index + 1 < n) {
        if (d_arr[index] > d_arr[index + 1]) {
            // Swap the two elements.
            int temp = d_arr[index];
            d_arr[index] = d_arr[index + 1];
            d_arr[index + 1] = temp;
        }
    }
}

int main() {
    const int n = 10;
    // Example unsorted array.
    int h_arr[n] = {64, 25, 12, 22, 11, 50, 30, 45, 15, 5};
    
    int *d_arr;
    size_t size = n * sizeof(int);
    
    // Allocate device memory for the array.
    cudaMalloc((void**)&d_arr, size);
    // Copy input data from host to device.
    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);
    
    // Define execution configuration.
    // We assume that one thread will handle one pair.
    // Maximum number of pairs in one phase is n/2.
    int threadsPerBlock = 256;
    int blocksPerGrid = ( (n/2) + threadsPerBlock - 1) / threadsPerBlock;
    
    // Odd-even transposition sort requires at most n phases.
    for (int phase = 0; phase < n; phase++) {
        // Alternate between even phase (phase % 2 == 0) and odd phase (phase % 2 == 1)
        oddEvenSortPhase<<<blocksPerGrid, threadsPerBlock>>>(d_arr, n, phase % 2);
        // Wait for the kernel to complete before starting the next phase.
        cudaDeviceSynchronize();
    }
    
    // Copy the sorted array back to host.
    cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
    
    // Print the sorted array.
    printf("Sorted Array:\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");
    
    // Clean up device memory.
    cudaFree(d_arr);
    
    return 0;
}
