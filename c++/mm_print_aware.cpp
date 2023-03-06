/* 
Cache-aware matrix multiplication in C++ with dynamic memory allocation
and manual control of the associated memory.

The principal source of the slowdown in naive matrix multiplication
for large matrices is the memory hierarchy present in most computer systems.
When a memory location is loaded, nearby areas are also placed in the cache,
a region of memory with fast access.
As such, our code should exploit spatial locality and work in one memory
location at a time, performing all the computation we can before sending
an instruction that will generate a cache miss.

This principle extends to GPU computation as well.
The memory hierarchy is so fundamental to computer systems that it seems
appropriate to enshrine "tiling" as a component of a domain-specific language
that produces efficient machine code for CPUs and GPUs.
This problem is solved (currently only for GPUs) by the Triton language.
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

void tile_mult(int* ptr_A_tile, int* ptr_B_tile, int* ptr_C_tile, int col, int hid, int BLOCK_SIZE){
    /*
    Multiplication of two BLOCK_SIZE tiles of A and B into C.
    */
    for(int i = 0; i < BLOCK_SIZE; i++){
        for(int j = 0; j < BLOCK_SIZE; j++){
            for(int k = 0; k < BLOCK_SIZE; k++){
                ptr_C_tile[i * col + j] += ptr_A_tile[i * hid + k] * ptr_B_tile[k * col + j];
            }
        }
    }
}

int* aware(int* ptr_A, int* ptr_B, int row, int col, int hid, int BLOCK_SIZE){
    /* Memory safety */
    assert(BLOCK_SIZE > 0 && row > BLOCK_SIZE && col > BLOCK_SIZE && hid > BLOCK_SIZE);
    assert(row % BLOCK_SIZE == 0 && col % BLOCK_SIZE == 0 and hid % BLOCK_SIZE == 0);

    int* ptr_C;
    int row_block = row/BLOCK_SIZE;
    int col_block = col/BLOCK_SIZE;
    int hid_block = hid/BLOCK_SIZE;
    int start_A, start_B, start_C;

    /* Allocate output matrix */
    ptr_C = (int*)malloc(sizeof(*ptr_C) * row * col);
    /* Zero entries of output matrix */
    for(int z1 = 0; z1 < row; z1++){
        for(int z2 = 0; z2 < col; z2++){
            ptr_C[z1 * col + z2] = 0;
        }
    }
    
    /* Cache-aware matrix multiplication by tiling */
    for(int ib = 0; ib < row_block; ib++){
        for(int jb = 0; jb < col_block; jb++){
            /* 
            Tile (ib, jb) of C consists of 
            ib * BLOCK_SIZE < row_index < (ib+1) * BLOCK_SIZE ,
            jb * BLOCK_SIZE < col_index < (jb+1) * BLOCK_SIZE 
            */
            for(int kb = 0; kb < hid_block; kb++){
                /*
                Multiply tile (ib, kb) of A with tile (kb, jb) of B
                */
                start_C = ib * BLOCK_SIZE * col + jb * BLOCK_SIZE;
                start_A = ib * BLOCK_SIZE * hid + kb * BLOCK_SIZE;
                start_B = kb * BLOCK_SIZE * col + jb * BLOCK_SIZE;
                tile_mult(&ptr_A[start_A], &ptr_B[start_B], &ptr_C[start_C], col, hid, BLOCK_SIZE);
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
    int BLOCK_SIZE;

    /* Command line argument to pass size of matrices and block */
    if(argc > 2){
        N = atoi(argv[1]);
        BLOCK_SIZE = atoi(argv[2]);
    }
    else if(argc == 2){ /* If no block size, use largest block */
        N = atoi(argv[1]);
        BLOCK_SIZE = N;
    }
    else{ /* Otherwise default to smallest nontrivial */
        N = 2;
        BLOCK_SIZE = 2;
    }
    /* Only ensure correctness and memory safety for divisible block sizes */
    assert(BLOCK_SIZE > 0 && N > BLOCK_SIZE && N % BLOCK_SIZE == 0);
    
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
    int* ptr_C = aware(ptr_A, ptr_B, N, N, N, BLOCK_SIZE);

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