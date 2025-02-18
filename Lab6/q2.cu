#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void parallelSelectionSortKernel(const int *d_in, int *d_out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int count = 0;
        int current = d_in[i];
        for (int j = 0; j < n; j++) {
            // Count an element as "smaller" if it is less than the current element.
            // For equal elements, use the index to break ties.
            if (d_in[j] < current || (d_in[j] == current && j < i)) {
                count++;
            }
        }
        // The count gives the sorted index for d_in[i].
        d_out[count] = current;
    }
}


int main() {
    const int n = 10;
    int h_arr[n] = {64, 25, 12, 22, 11, 50, 30, 45, 15, 5};

    int *d_in, *d_out;
    size_t size = n * sizeof(int);

    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);


    cudaMemcpy(d_in, h_arr, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    parallelSelectionSortKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, n);


    cudaDeviceSynchronize();

    int sorted_arr[n];
    cudaMemcpy(sorted_arr, d_out, size, cudaMemcpyDeviceToHost);

    printf("Sorted Array:\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", sorted_arr[i]);
    }
    printf("\n");

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
