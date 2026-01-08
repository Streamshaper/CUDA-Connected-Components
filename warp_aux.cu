#include "warp_aux.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <matio.h>

//============================= CUDA KERNELS ============================= //

__global__ void label_propagation_kernel(uint64_t* labels, int* active, int* next_active, 
                                        const uint64_t* indices, const uint64_t* ind_ptr, 
                                        uint64_t n_nodes, int* changed_flag)
{
    int warps_per_block = blockDim.x >> 5;
    int warp_in_block = threadIdx.x >> 5;
    int warp_id = blockIdx.x * warps_per_block + warp_in_block;
    int lane    = threadIdx.x & 31;
    int total_warps = gridDim.x * warps_per_block;

    if (warp_id >= total_warps) return;

    for (uint64_t node = warp_id; node < n_nodes; node += total_warps) {
        if (!active[node]) continue;
        uint64_t start = ind_ptr[node];
        uint64_t end   = ind_ptr[node + 1];
        if (start == end) continue;

        uint64_t local_min = UINT64_MAX;

        // Parallel neighbor traversal
        for (uint64_t k = start + lane; k < end; k += 32) {
            uint64_t nbr = indices[k];
            local_min = min(local_min, labels[nbr]);
        }

        // Warp reduction
        for (int offset = 16; offset > 0; offset >>= 1)
            local_min = min(local_min,
            __shfl_down_sync(0xffffffff, local_min, offset));

        // Lane 0 performs the update
        if (lane == 0 && local_min < labels[node]) {
            labels[node] = local_min;
            changed_flag = 1;   // No atomics

            // Wake neighbors
            for (uint64_t k = start; k < end; k++)
                next_active[indices[k]] = 1;    // No atomics
        }
    }

}

__global__ void reset_active_kernel(int* next_active, uint64_t n_nodes) 
{
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;
    next_active[i] = 0;
}


//============================== AUXILIARY ============================== //

void open_matrix (char* name, uint64_t** indices, uint64_t** ind_ptr, uint64_t* n_nodes)
{
    mat_t *matfp = Mat_Open(name, MAT_ACC_RDONLY);
    if (!matfp) { fprintf(stderr,"Cannot open file\n"); exit(2); }

    matvar_t *problem = Mat_VarRead(matfp, "Problem");
    if (!problem || problem->class_type != MAT_C_STRUCT) { fprintf(stderr,"Problem struct missing\n"); exit(2); }

    matvar_t *Avar = Mat_VarGetStructFieldByName(problem, "A", 0);
    if (!Avar || Avar->class_type != MAT_C_SPARSE) { fprintf(stderr,"A is not sparse\n"); exit(2); }

    mat_sparse_t *A = (mat_sparse_t*)Avar->data;
    size_t n = Avar->dims[1], nnz = A->nzmax;

    *indices = (uint64_t*)malloc (nnz*sizeof(uint64_t));
    *ind_ptr = (uint64_t*)malloc ((n+1)*sizeof(uint64_t));

    for (size_t q=0; q<nnz; q++)
        (*indices)[q] = (uint64_t)A->ir[q];

    for (size_t q=0; q<=n; q++)
        (*ind_ptr)[q] = (uint64_t)A->jc[q];
        
    *n_nodes = (uint64_t)n;

    Mat_VarFree(problem);
    Mat_Close(matfp);
    printf ("The graph has %lu nodes and %lu edges in total.\n", *n_nodes, (uint64_t)nnz/2);
}