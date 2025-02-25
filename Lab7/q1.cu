#include<stdio.h>
#include<string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void CountWord(const char *sentence, int sentence_len, const char *word, int word_len, int *count){
    int idx = threadIdx.x+ blockDim.x*blockIdx.x;
    bool match = true;
    if (idx<sentence_len-word_len){
        match = true;
    }
    for (int j=0;j<word_len;j++){
        if(sentence[idx + j]!=word[j]){
            match = false;
            break;
        }
    }
    if (match){
        atomicAdd(count,1);
    }
}
int main(){
    const char *h_sentence = "This is a sample sentence to sample word occurences";
    const char *h_word = "sample";
    int senLen = strlen(h_sentence);
    int wordLen = strlen(h_word);
    char *d_sentence;
    char *d_word;
    int *d_count;
    int h_count = 0;

    cudaMalloc((void**)&d_sentence,senLen*sizeof(char));
    cudaMalloc((void**)&d_word,wordLen*sizeof(char));
    cudaMalloc((void**)&d_count,sizeof(int));

    cudaMemcpy(d_sentence,h_sentence,(senLen)*sizeof(char),cudaMemcpyHostToDevice);
    cudaMemcpy(d_word,h_word,(wordLen)*sizeof(char),cudaMemcpyHostToDevice);
    cudaMemcpy(d_count,&h_count,sizeof(int),cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (threadsPerBlock+senLen-1)/(threadsPerBlock);

    CountWord<<<blocksPerGrid,threadsPerBlock>>>(d_sentence,senLen,d_word,wordLen,d_count);

    cudaMemcpy(&h_count,d_count,sizeof(int),cudaMemcpyDeviceToHost);

    printf("The word %s occurred %d times in the sentence.\n",h_word,h_count);

    cudaFree(d_sentence);
    cudaFree(d_word);
    cudaFree(d_count);
    return 0;


}