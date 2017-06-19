#!/bin/sh
export TF_INC=`python -c "import tensorflow; print(tensorflow.sysconfig.get_include())"`
nvcc -std=c++11 -c -o correlation_gpu.cu.o correlation_gpu.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
nvcc -std=c++11 -c -o correlation_grad_gpu.cu.o correlation_grad_gpu.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

