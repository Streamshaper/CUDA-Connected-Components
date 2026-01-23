#include "warp_aux.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <matio.h>

//============================= CUDA KERNELS ============================= //

__global__ void label_propagation_kernel(uint32_t* labels, int* active, int* next_active,
                                        const uint32_t* indices, const uint32_t* ind_ptr,
                                        uint32_t n_nodes, int* changed_flag)
{
    int warps_per_block = blockDim.x >> 5;
    int warp_in_block = threadIdx.x >> 5;
    int warp_id = blockIdx.x * warps_per_block + warp_in_block;
    int lane    = threadIdx.x & 31;
    int total_warps = gridDim.x * warps_per_block;

    if (warp_id >= total_warps) return;

    for (uint32_t node = warp_id; node < n_nodes; node += total_warps) {
        if (!active[node]) continue;
        uint32_t start = ind_ptr[node];
        uint32_t end   = ind_ptr[node + 1];
        if (start == end) continue;

        uint32_t local_min = UINT32_MAX;

        // Parallel neighbor traversal
        for (uint32_t k = start + lane; k < end; k += 32) {
            uint32_t nbr = indices[k];
            local_min = min(local_min, labels[nbr]);
        }

        // Warp reduction
        for (int offset = 16; offset > 0; offset >>= 1)
            local_min = min(local_min,
            __shfl_down_sync(0xffffffff, local_min, offset));

        // Lane 0 performs the update
        if (lane == 0 && local_min < labels[node]) {
            labels[node] = local_min;
            *changed_flag = 1;   // No atomics

            // Wake neighbors
            for (uint32_t k = start; k < end; k++)
                next_active[indices[k]] = 1;    // No atomics
        }
    }

}

__global__ void initialize_kernel (uint32_t* labels, uint8_t* active, uint8_t *next_active, uint32_t n_nodes)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;
    labels[i] = i;  // initial label
    active[i] = 1;
    next_active[i] = 0;
}

__global__ void reset_active_kernel(uint8_t* next_active, uint32_t n_nodes)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;
    next_active[i] = 0;
}


//============================== AUXILIARY ============================== //

void open_matrix (char* name, uint32_t** indices, uint32_t** ind_ptr, uint32_t* n_nodes,
                mat_t** matfp_out, matvar_t** problem_out)
{
    mat_t *matfp = Mat_Open(name, MAT_ACC_RDONLY);
    if (!matfp) { fprintf(stderr,"Cannot open file\n"); exit(2); }

    matvar_t *problem = Mat_VarRead(matfp, "Problem");
    if (!problem || problem->class_type != MAT_C_STRUCT) { fprintf(stderr,"Problem struct missing\n"); exit(2); }

    matvar_t *Avar = Mat_VarGetStructFieldByName(problem, "A", 0);
    if (!Avar || Avar->class_type != MAT_C_SPARSE) { fprintf(stderr,"A is not sparse\n"); exit(2); }

    mat_sparse_t *A = (mat_sparse_t*)Avar->data;
    size_t n = Avar->dims[1], nnz = A->nzmax;

    printf ("Opened file successfully!\n");
    *indices = (uint32_t*)A->ir;
    *ind_ptr = (uint32_t*)A->jc;
    *n_nodes = (uint32_t)n;

    *matfp_out = matfp;
    *problem_out = problem;
    printf ("The graph has %u nodes and %lu edges in total.\n", *n_nodes, (uint64_t)nnz/2);
}
