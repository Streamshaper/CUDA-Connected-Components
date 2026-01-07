#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <matio.h>

#define CUDA_CHECK() \
do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        printf("CUDA error %s at %s:%d\n", \
               cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)


// Graph data
uint64_t n_nodes;
uint64_t* indices;
uint64_t* ind_ptr;
char* matrix_name;

// ----------------------- CUDA Kernels -----------------------
__global__ void label_propagation_kernel(
    uint64_t* labels,
    int* active,
    int* next_active,
    const uint64_t* indices,
    const uint64_t* ind_ptr,
    uint64_t n_nodes,
    int* changed_flag)
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
            atomicExch(changed_flag, 1);

            // Wake neighbors
            for (uint64_t k = start; k < end; k++)
                atomicExch(&next_active[indices[k]], 1);
        }
    }

}

__global__ void reset_active_kernel(int* next_active, uint64_t n_nodes) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;
    next_active[i] = 0;
}

// AUX
void open_matrix (char* name)
{
    mat_t *matfp = Mat_Open(name, MAT_ACC_RDONLY);
    if (!matfp) { fprintf(stderr,"Cannot open file\n"); exit(2); }

    matvar_t *problem = Mat_VarRead(matfp, "Problem");
    if (!problem || problem->class_type != MAT_C_STRUCT) { fprintf(stderr,"Problem struct missing\n"); exit(2); }

    matvar_t *Avar = Mat_VarGetStructFieldByName(problem, "A", 0);
    if (!Avar || Avar->class_type != MAT_C_SPARSE) { fprintf(stderr,"A is not sparse\n"); exit(2); }

    mat_sparse_t *A = (mat_sparse_t*)Avar->data; // Correct way to access sparse data
    size_t n = Avar->dims[1], nnz = A->nzmax;

    indices = (uint64_t*)malloc (nnz*sizeof(uint64_t));
    ind_ptr = (uint64_t*)malloc ((n+1)*sizeof(uint64_t));

    for (size_t q=0; q<nnz; q++)
        indices[q] = (uint64_t)A->ir[q];

    for (size_t q=0; q<=n; q++)
        ind_ptr[q] = (uint64_t)A->jc[q];
        
    n_nodes = (uint64_t)n;

    Mat_VarFree(problem);
    Mat_Close(matfp);
    printf ("The graph has %lu nodes and %lu edges in total.\n", n_nodes, (uint64_t)nnz/2);
}

// ----------------------- Main Function -----------------------
int main(int argc, char* argv[]) {

    if (argc > 1)
        matrix_name = argv[1];
    else{
        printf ("Missing argument: matrix_name\n");
        exit (0);
    }
    
    int threads = 256;
    if (argc > 2 && atoi(argv[2])<8192 && atoi(argv[2])>1)	 
    	threads = atoi(argv[2]);
    threads = (threads / 32) * 32;
    if (threads == 0) threads = 32;


    printf("Running with %d threads (%d warps per block)\n", threads, threads / 32);

    open_matrix (matrix_name);

    uint64_t* labels = (uint64_t*)malloc(n_nodes * sizeof(uint64_t));
    int* active = (int*)malloc(n_nodes * sizeof(int));
    int* next_active = (int*)malloc(n_nodes * sizeof(int));
    for (uint64_t i = 0; i < n_nodes; i++) {
        labels[i] = i;  // initial label
        active[i] = 1;
        next_active[i] = 0;
    }

    // --- Device arrays ---
    uint64_t *d_labels, *d_indices, *d_ind_ptr;
    int *d_active, *d_next_active;
    int *d_changed_flag, h_changed_flag;

    cudaMalloc(&d_labels, n_nodes * sizeof(uint64_t));
    cudaMalloc(&d_active, n_nodes * sizeof(int));
    cudaMalloc(&d_next_active, n_nodes * sizeof(int));
    cudaMalloc(&d_indices, ind_ptr[n_nodes] * sizeof(uint64_t));
    cudaMalloc(&d_ind_ptr, (n_nodes + 1) * sizeof(uint64_t));
    cudaMalloc(&d_changed_flag, sizeof(int));

    cudaMemcpy(d_labels, labels, n_nodes * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_active, active, n_nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_next_active, next_active, n_nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, indices, ind_ptr[n_nodes] * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ind_ptr, ind_ptr, (n_nodes + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // --- Launch parameters ---
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int blocks = 4 * prop.multiProcessorCount;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --- Iterative label propagation ---
    cudaEventRecord(start); 
    int iteration = 0;
    do {
        iteration++;
        h_changed_flag = 0;
        cudaMemcpy(d_changed_flag, &h_changed_flag, sizeof(int), cudaMemcpyHostToDevice);

        label_propagation_kernel<<<blocks, threads>>> (d_labels, d_active, 
            d_next_active, d_indices, d_ind_ptr, n_nodes, d_changed_flag);

        CUDA_CHECK ();
        cudaDeviceSynchronize();
        CUDA_CHECK ();

        cudaMemcpy(&h_changed_flag, d_changed_flag, sizeof(int), cudaMemcpyDeviceToHost);

        // Swap active arrays
        int* temp = d_active;
        d_active = d_next_active;
        d_next_active = temp;

        // Reset next_active
        int reset_blocks = (n_nodes + threads - 1) / threads;
        reset_active_kernel<<<reset_blocks, threads>>>(d_next_active, n_nodes);

        CUDA_CHECK ();
        cudaDeviceSynchronize();
        CUDA_CHECK ();

    } while (h_changed_flag);

    // --- Copy labels back to host ---
    cudaMemcpy(labels, d_labels, n_nodes * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // --- Count unique labels (serial) ---
    uint8_t* found = (uint8_t*)calloc(n_nodes, sizeof(uint8_t));
    uint64_t components = 0;
    for (uint64_t i = 0; i < n_nodes; i++) {
        if (!found[labels[i]]) {
            found[labels[i]] = 1;
            components++;
        }
    }
    printf("Connected Components: %lu\n", components);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU kernel time: %.3f ms\n", ms);

    // --- Cleanup ---
    cudaFree(d_labels);
    cudaFree(d_active);
    cudaFree(d_next_active);
    cudaFree(d_indices);
    cudaFree(d_ind_ptr);
    cudaFree(d_changed_flag);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(labels);
    free(active);
    free(next_active);
    free(found);
    free(indices);
    free(ind_ptr);


    return 0;
}
