/* 
Naive matrix multiplication in C++ with dynamic memory allocation
and manual control of the associated memory.

This version prints the matrices and is intended for testing.
Pipe the output to a textfile and copy to Mathematica.
Command line: ./mm_print {N} > out.txt
*/

#include <cstdlib>
#include <cassert>
#include <iostream>
using namespace std;

void print_matrix(int* ptr, int row, int col){
    string c;

    cout << "{";
    
    for(int i = 0; i < row; i++){
        cout << "{";
        for(int j = 0; j < col; j++){
            if(j < col-1){
                c = ",";
            }
            else{
                c = "}";
            }
            cout << ptr[i * col + j] << c;
        }
        if(i < row-1){
            cout << ",";
            cout << endl;
        }
    }
    
    cout << "}\n\n";
}

int* blind(int* ptr_A, int* ptr_B, int row, int col, int hid){
    int* ptr_C;
    /* Allocate output matrix */
    ptr_C = (int*)malloc(sizeof(*ptr_C) * row * col);
    
    /* Naive matrix multiplication */
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            for(int k = 0; k < hid; k++){
                ptr_C[i * col + j] += ptr_A[i * hid + k] * ptr_B[k * col + j];
            }
        }
    }

    return ptr_C;
}

int main(int argc, const char* argv[]){
    /* 
    Test matrix multiplication on N x N square matrices 
    with random integer entries between 0 and rand_max.
    */
    int rand_max = 5;
    int N;

    /* Command line argument to pass size of matrices */
    if(argc > 1){
        N = atoi(argv[1]);
    }
    else{ /* Otherwise default to smallest nontrivial */
        N = 2;
    }
    assert(N > 0);
    
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

    /* Matrix multiplication */
    int* ptr_C = blind(ptr_A, ptr_B, N, N, N);

    /* Check correctness visually, testing. */
    print_matrix(ptr_A, N, N);
    print_matrix(ptr_B, N, N);
    print_matrix(ptr_C, N, N);

    /* Free memory allocations */
    free(ptr_A);
    free(ptr_B);
    free(ptr_C);

    return 0;
}