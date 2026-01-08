#ifndef WARPING_H_INCLUDED
#define WARPING_H_INCLUDED

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
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

__global__ void label_propagation_kernel(uint64_t* labels, int* active, int* next_active, 
                                        const uint64_t* indices, const uint64_t* ind_ptr, 
                                        uint64_t n_nodes, int* changed_flag);

__global__ void reset_active_kernel(int* next_active, uint64_t n_nodes);

void open_matrix (char* name, uint64_t* indices, uint64_t* ind_ptr, uint64_t* n_nodes);

#endif