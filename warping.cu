#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <matio.h>

#include "warp_aux.h"

// Graph data
uint64_t n_nodes;
uint64_t* indices;
uint64_t* ind_ptr;
char* matrix_name;

size_t free_m, total;


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


    printf("Running with %d threads (%d warps) per block\n", threads, threads / 32);

    open_matrix (matrix_name, &indices, &ind_ptr, &n_nodes);

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

    cudaMemGetInfo(&free_m, &total);
    printf("GPU memory used: %.2f/%.2f MB\n", (total/1e6)-(free_m/1e6), total/1e6);

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
    printf("Connected Components: %lu, found in %d iterations\n", components, iteration);
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
