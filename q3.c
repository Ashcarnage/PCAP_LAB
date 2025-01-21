#include<stdio.h>
#include"mpi.h"
#include<ctype.h>
#include<string.h>
#include<stdlib.h>

int check_vowel(char c){
    char lower_c = tolower(c);
    return (lower_c>='a'&& lower_c<='z'&& !(lower_c == 'a' || lower_c == 'e' || lower_c == 'i' || lower_c == 'o' || lower_c == 'u'));
}

int main(int argc, char *argv[]){
    int rank,size;
    char *input_str = NULL;
    int str_len= 0;
    int *vowels_per_process = NULL;
    int total_non_vowels =  0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0){
        printf("Enter the string of length %d\n",size);
        input_str = (char*)malloc(1000*sizeof(char));
        scanf("%s",input_str);
        str_len = strlen(input_str);
        if (str_len % size!=0){
            printf("ERROR: String length os not divisible by the number of processes");
        }
    }

    MPI_Bcast(&str_len,1,MPI_INT,0,MPI_COMM_WORLD);
    int local_len = str_len/size;
    // int non_vowels = (int*)malloc(local_len*sizeof(int));
    char *local_str = (char*)malloc(local_len*sizeof(char));
    MPI_Scatter(input_str, local_len, MPI_CHAR, local_str, local_len, MPI_CHAR, 0, MPI_COMM_WORLD);
    int local_count = 0;
    for (int i=0;i<local_len;i++){
        if(check_vowel(local_str[i])){
            local_count+=1;
        }
    }
    if(rank == 0){
        vowels_per_process = (int*)malloc(sizeof(int)*size);
    }
    MPI_Gather(&local_count,1,MPI_INT,vowels_per_process,1,MPI_INT,0,MPI_COMM_WORLD);

    if (rank == 0){
        printf("Non-vowels found per process : \n");
        for(int i=0;i<size;i++){
            int val = vowels_per_process[i];
            printf("Process : %d found %d Non-Vowels\n",i,val);
            total_non_vowels+=val;
        }
        printf("Total Number of non-vowels : %d",total_non_vowels);
        free(input_str);
        free(vowels_per_process);
    }
    free(local_str);
    MPI_Finalize();
    return 0 ;

}