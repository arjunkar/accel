/* 
Matrix multiplication in C++ with dynamic memory allocation
and manual control of the associated memory.
*/

#include <cstdlib>
#include <iostream>
using namespace std;

void print_matrix(int* ptr, int row, int col){
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            cout << ptr[i * col + j] << " ";
        }
        cout << endl;
    }
    cout << "\n";
}

int* blind(int* ptr_A, int* ptr_B, int row, int col, int hid){
    int* ptr_C;

    ptr_C = (int*)malloc(sizeof(*ptr_C) * row * col);
    
    /* Naive matrix multiplication */
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            ptr_C[i * col + j] = 0;
            for(int k = 0; k < hid; k++){
                ptr_C[i * col + j] += ptr_A[i * hid + k] * ptr_B[k * col + j];
            }
        }
    }

    return ptr_C;
}

int main(int argc, const char* argv[]){
    int rand_max = 5;
    int N;

    /* Command line argument to pass size of matrices */
    if(argc > 1){
        N = atoi(argv[1]);
    }
    else{ /* Otherwise default to smallest nontrivial */
        N = 2;
    }
    
    /* A cast to (int*) is needed in C++ malloc */
    int* ptr_A = (int*)malloc(sizeof(int) * N * N);
    int* ptr_B = (int*)malloc(sizeof(int) * N * N);

    /* Fill arrays with random integers 0 <= x < N */
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            ptr_A[i * N + j] = rand() % rand_max;
            ptr_B[i * N + j] = rand() % rand_max;
        }
    }

    int* ptr_C = blind(ptr_A, ptr_B, N, N, N);

    /* 
    Check correctness visually, testing. 
    mm_print compiled with these uncommented.
    */

    /*
    print_matrix(ptr_A, N, N);
    print_matrix(ptr_B, N, N);
    print_matrix(ptr_C, N, N);
    */
    
    /* Free memory allocations */
    free(ptr_A);
    free(ptr_B);
    free(ptr_C);

    return 0;
}