# CUDA-Connected-Components
The code on this repository was tested and benchmarked on Google Colab. 

### Purpose
The code contained here is a parallelized implementation of a label propagation algorithm that calculates the number of connected components in a graph. Each graph is represented by its adjacency matrix, which is stored in Compressed Sparse Row format. The design is written in CUDA, allowing it to be run efficiently on NVIDIA GPUs. There are three variations of the program:  
* `lead.cu`: One thread per node
* `warping.cu`: One warp per node
* `blocking.cu`: One block per node

### Dependencies
* CUDA Toolkit
* libmatio-dev (MATIO library)
* libhdf5-dev (HDF5)
* zlib1g-dev

### Running the code
Each executable receives two arguments: the name of the matrix to be processed and the number of threads to be used per block.
Example code for a Google Colab notebook, running an NVIDIA T4 GPU:  
```
!apt-get update
!apt-get install -y libhdf5-dev libmatio-dev zlib1g-dev
!git clone https://github.com/Streamshaper/CUDA-Connected-Components
%cd ./CUDA-Connected-Components
!wget https://suitesparse-collection-website.herokuapp.com/mat/SNAP/com-LiveJournal.mat
!wget https://suitesparse-collection-website.herokuapp.com/mat/SNAP/com-Orkut.mat
!wget https://suitesparse-collection-website.herokuapp.com/mat/MAWI/mawi_201512020330.mat
```
```
!nvcc -O3 lead.cu -o lead \
  -arch=sm_75 \
  -I/usr/include \
  -L/usr/lib/x86_64-linux-gnu \
  -lmatio -lhdf5_serial -lz
!nvcc -O3 warp_aux.cu warping.cu -o warping \
  -arch=sm_75 \
  -I/usr/include \
  -L/usr/lib/x86_64-linux-gnu \
  -lmatio -lhdf5_serial -lz
!nvcc -O3 warp_aux.cu blocking.cu -o blocking \
  -arch=sm_75 \
  -I/usr/include \
  -L/usr/lib/x86_64-linux-gnu \
  -lmatio -lhdf5_serial -lz
```
```
!./lead com-LiveJournal.mat 256
!./warping mawi_201512020330.mat 64
!./blocking com-Orkut.mat 128
```
