#!/bin/sh
export TF_INC=/usr/local/lib/python2.7/dist-packages/tensorflow/include
nvcc -std=c++11 -c -o correlation_gpu.cu.o correlation_gpu.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
nvcc -std=c++11 -c -o correlation_grad_gpu.cu.o correlation_grad_gpu.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

