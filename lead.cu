#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <matio.h>

// Graph data (CSR format)
uint64_t n_nodes;
uint64_t* indices;
uint64_t* ind_ptr;

// ----------------------- CUDA Kernels -----------------------
__global__ void label_propagation_kernel(
    uint64_t* labels,
    int* active,
    int* next_active,
    uint64_t* indices,
    uint64_t* ind_ptr,
    uint64_t n_nodes,
    int* changed_flag)
{
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;
    if (!active[i]) return;

    uint64_t start = ind_ptr[i];
    uint64_t end   = ind_ptr[i + 1];
    if (start == end) return;

    uint64_t min_label = labels[i];
    for (uint64_t k = start; k < end; k++) {
        uint64_t neighbor = indices[k];
        if (labels[neighbor] < min_label)
            min_label = labels[neighbor];
    }

    if (min_label < labels[i]) {
        labels[i] = min_label;
        *changed_flag = 1;  // signal host that a change occurred

        // Mark neighbors as active for next iteration
        for (uint64_t k = start; k < end; k++) {
            atomicExch(&next_active[indices[k]], 1);
        }
    }
}

__global__ void reset_active_kernel(uint8_t* next_active, uint64_t n_nodes) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;
    next_active[i] = 0;
}

// AUX
void open_matrix (const char* name)
{
    mat_t *matfp = Mat_Open(name, MAT_ACC_RDONLY);
    if (!matfp) { fprintf(stderr,"Cannot open file\n"); exit(2); }

    matvar_t *problem = Mat_VarRead(matfp, "Problem");
    if (!problem || problem->class_type != MAT_C_STRUCT) { fprintf(stderr,"Problem struct missing\n"); exit(2); }

    matvar_t *Avar = Mat_VarGetStructFieldByName(problem, "A", 0);
    if (!Avar || Avar->class_type != MAT_C_SPARSE) { fprintf(stderr,"A is not sparse\n"); exit(2); }

    mat_sparse_t *A = (mat_sparse_t*)Avar->data; // Correct way to access sparse data
    size_t m = Avar->dims[0], n = Avar->dims[1], nnz = A->nzmax;

    indices = (uint64_t*)malloc (nnz*sizeof(uint64_t));
    ind_ptr = (uint64_t*)malloc ((n+1)*sizeof(uint64_t));

    for (size_t q=0; q<nnz; q++)
        indices[q] = (uint64_t)A->ir[q];

    for (size_t q=0; q<=n; q++)
        ind_ptr[q] = (uint64_t)A->jc[q];
        
    n_nodes = (uint64_t)n;

    Mat_VarFree(problem);
    Mat_Close(matfp);
        printf ("The graph has %llu nodes and %llu edges in total.\n", n_nodes, (uint64_t)nnz/2);

}

// ----------------------- Main Function -----------------------
int main() {

    open_matrix ("com-LiveJournal.mat");

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
    cudaMalloc(&d_active, n_nodes * sizeof(uint8_t));
    cudaMalloc(&d_next_active, n_nodes * sizeof(uint8_t));
    cudaMalloc(&d_indices, ind_ptr[n_nodes] * sizeof(uint64_t));
    cudaMalloc(&d_ind_ptr, (n_nodes + 1) * sizeof(uint64_t));
    cudaMalloc(&d_changed_flag, sizeof(int));

    cudaMemcpy(d_labels, labels, n_nodes * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_active, active, n_nodes * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_next_active, next_active, n_nodes * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, indices, ind_ptr[n_nodes] * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ind_ptr, ind_ptr, (n_nodes + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // --- Launch parameters ---
    int threads = 256;
    int blocks = (n_nodes + threads - 1) / threads;

    // --- Iterative label propagation ---
    int iteration = 0;
    do {
        iteration++;
        h_changed_flag = 0;
        cudaMemcpy(d_changed_flag, &h_changed_flag, sizeof(int), cudaMemcpyHostToDevice);

        label_propagation_kernel<<<blocks, threads>>>(
            d_labels, d_active, d_next_active, d_indices, d_ind_ptr, n_nodes, d_changed_flag
        );
        cudaDeviceSynchronize();

        cudaMemcpy(&h_changed_flag, d_changed_flag, sizeof(int), cudaMemcpyDeviceToHost);

        // Swap active arrays
        int* temp = d_active;
        d_active = d_next_active;
        d_next_active = temp;

        // Reset next_active
        reset_active_kernel<<<blocks, threads>>>(d_next_active, n_nodes);
        cudaDeviceSynchronize();

    } while (h_changed_flag);

    // --- Copy labels back to host ---
    cudaMemcpy(labels, d_labels, n_nodes * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // --- Count unique labels (serial) ---
    uint8_t* found = (uint8_t*)calloc(n_nodes, sizeof(uint8_t));
    uint64_t components = 0;
    for (uint64_t i = 0; i < n_nodes; i++) {
        if (!found[labels[i]]) {
            found[labels[i]] = 1;
            components++;
        }
    }
    printf("Connected Components: %llu\n", components);

    // --- Cleanup ---
    cudaFree(d_labels);
    cudaFree(d_active);
    cudaFree(d_next_active);
    cudaFree(d_indices);
    cudaFree(d_ind_ptr);
    cudaFree(d_changed_flag);

    free(labels);
    free(active);
    free(next_active);
    free(found);

    return 0;
}
