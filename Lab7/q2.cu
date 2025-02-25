#include<stdio.h>
#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void generateRs(char *S,char *RS,int lenS, int lenRS){
    int idx= threadIdx.x + blockDim.x*blockIdx.x;
    if(idx<lenRS){
        RS[idx] = S[idx%lenS];
    }
}

int main(){
    char *h_S = "PCAP";
    int lenS = strlen(h_S);
    int lenRS = 10;
    char *h_RS = (char*)malloc(sizeof(char)*10);


    char *d_S,*d_RS;
    cudaMalloc((void **)&d_S,lenS*sizeof(char));
    cudaMalloc((void **)&d_RS,lenRS*sizeof(char));
    cudaMemcpy(d_S,h_S,lenS*sizeof(char),cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridsize = (blockSize + lenRS -1)/blockSize;
    generateRs<<<gridsize,blockSize>>>(d_S,d_RS,lenS,lenRS);

    cudaMemcpy(h_RS,d_RS,lenRS*sizeof(char),cudaMemcpyDeviceToHost);
    // h_RS[lenRS]='\0';
    printf("The string %s is converted into ---> %s\n",h_S,h_RS);



}