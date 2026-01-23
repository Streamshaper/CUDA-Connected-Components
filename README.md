# CUDA-Connected-Components
The code on this repository was tested and benchmarked on Google Colab. 
### Running the code
Example code for a Google Colab notebook:  
```
!apt-get update
!apt-get install -y libhdf5-dev libmatio-dev
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
