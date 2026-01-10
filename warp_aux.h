#ifndef WARP_AUX_H_INCLUDED
#define WARP_AUX_H_INCLUDED

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <matio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK() \
do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        printf("CUDA error %s at %s:%d\n", \
               cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

__global__ void label_propagation_kernel(int* labels, int* active, int* next_active, 
                                        const int* indices, const int* ind_ptr, 
                                        int n_nodes, int* changed_flag);

__global__ void initialize_kernel (int* labels, int* active, int *next_active, int n_nodes);

__global__ void reset_active_kernel(int* next_active, int n_nodes);

void open_matrix (char* name, int** indices, int** ind_ptr, int* n_nodes,
                mat_t** matfp_out, matvar_t** problem_out);

#endif