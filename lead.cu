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
uint32_t n_nodes;
uint32_t* indices;
uint32_t* ind_ptr;
char* matrix_name;

// ----------------------- CUDA Kernels -----------------------

__global__ void initialize_kernel (uint32_t* labels, int* active, int *next_active, uint32_t n_nodes)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;
    labels[i] = i;  // initial label
    active[i] = 1;
    next_active[i] = 0;
}

__global__ void label_propagation_kernel(
    uint32_t* labels,
    int* active,
    int* next_active,
    uint32_t* indices,
    uint32_t* ind_ptr,
    uint32_t n_nodes,
    int* changed_flag)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;
    if (!active[i]) return;

    uint32_t start = ind_ptr[i];
    uint32_t end   = ind_ptr[i + 1];
    if (start == end) return;

    uint32_t min_label = labels[i];
    for (uint32_t k = start; k < end; k++) {
        uint32_t neighbor = indices[k];
        if (labels[neighbor] < min_label)
            min_label = labels[neighbor];
    }

    if (min_label < labels[i]) {
        labels[i] = min_label;
        atomicExch(changed_flag, 1);  // signal host that a change occurred

        // Mark neighbors as active for next iteration
        for (uint32_t k = start; k < end; k++) {
            atomicExch(&next_active[indices[k]], 1);
        }
    }
}

__global__ void reset_active_kernel(int* next_active, uint32_t n_nodes) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;
    next_active[i] = 0;
}

// Auxiliary Function
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
    printf ("The graph has %u nodes and %u edges in total.\n", *n_nodes, (uint64_t)nnz/2);
}

// ----------------------- Main Function -----------------------
int main(int argc, char *argv[]) {

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

    mat_t    *matfp   = NULL;
    matvar_t *problem = NULL;
    open_matrix (matrix_name, &indices, &ind_ptr, &n_nodes, &matfp, &problem);

    uint32_t* labels = (uint32_t*)malloc(n_nodes * sizeof(uint32_t));

    // Device arrays
    uint32_t *d_labels, *d_indices, *d_ind_ptr;
    int *d_active, *d_next_active;
    int *d_changed_flag, h_changed_flag;

    int re_set_blocks = (n_nodes + threads - 1) / threads;

    cudaMalloc(&d_labels, n_nodes * sizeof(uint32_t));
    cudaMalloc(&d_active, n_nodes * sizeof(int));
    cudaMalloc(&d_next_active, n_nodes * sizeof(int));
    cudaMalloc(&d_indices, ind_ptr[n_nodes] * sizeof(uint32_t));
    cudaMalloc(&d_ind_ptr, (n_nodes + 1) * sizeof(uint32_t));
    cudaMalloc(&d_changed_flag, sizeof(int));

    cudaMemcpy(d_indices, indices, ind_ptr[n_nodes] * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ind_ptr, ind_ptr, (n_nodes + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
    Mat_VarFree(problem);
    Mat_Close(matfp);

    initialize_kernel <<<re_set_blocks, threads>>> (d_labels, d_active, d_next_active, n_nodes);

    int blocks = (n_nodes + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Iterative label propagation
    cudaEventRecord(start); 
    int iteration = 0;
    do {    // Iterative label propagation
        iteration++;
        h_changed_flag = 0;
        cudaMemcpy(d_changed_flag, &h_changed_flag, sizeof(int), cudaMemcpyHostToDevice);

        label_propagation_kernel<<<blocks, threads>>>(
            d_labels, d_active, d_next_active, d_indices, d_ind_ptr, n_nodes, d_changed_flag
        );

        CUDA_CHECK ();
        cudaDeviceSynchronize();
        CUDA_CHECK ();

        cudaMemcpy(&h_changed_flag, d_changed_flag, sizeof(int), cudaMemcpyDeviceToHost);

        // Swap active arrays
        int* temp = d_active;
        d_active = d_next_active;
        d_next_active = temp;

        // Reset next_active
        reset_active_kernel<<<re_set_blocks, threads>>>(d_next_active, n_nodes);

        CUDA_CHECK ();
        cudaDeviceSynchronize();
        CUDA_CHECK ();

    } while (h_changed_flag);

    // Copy labels back to host
    uint32_t* labels = (uint32_t*)malloc(n_nodes * sizeof(uint32_t));
    cudaMemcpy(labels, d_labels, n_nodes * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Count unique labels (serial)
    uint8_t* found = (uint8_t*)calloc(n_nodes, sizeof(uint8_t));
    uint32_t components = 0;
    for (uint32_t i = 0; i < n_nodes; i++) {
        if (!found[labels[i]]) {
            found[labels[i]] = 1;
            components++;
        }
    }
    printf("Connected Components: %u\n", components);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU kernel time: %.3f ms\n", ms);

    // Cleanup
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

    return 0;
}
