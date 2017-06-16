/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

__global__ void CorrelationGradKernel(const float* a, const float*b,const float*grad,float* out_a,float*out_b, const int batch_size,const int num_rows, const int num_cols, const int depth,const int num_offsets, const int* offset_list)  {
    int one_d_size   = depth;
    int two_d_size   = one_d_size*num_cols;
    int three_d_size = two_d_size*num_rows;

    int out1 = num_offsets;
    int out2 = num_cols * out1;
    int out3 = num_rows * out2;

    for (int i = 0; i < batch_size; i++) {
        for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < num_rows; j += blockDim.x * gridDim.x) {
          for (int k = 0; k < num_cols; k++) {
              int a_root = three_d_size*i + two_d_size*j+one_d_size * k;
              for (int m =0; m < depth; m++ ) {
                    out_a[a_root+m] = 0;
                    out_b[a_root+m] = 0;
              }
          }
        }
    }

    for (int i = 0; i < batch_size; i++) {
        int b_root = i*three_d_size;
        for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < num_rows; j += blockDim.x * gridDim.x) {
          for (int k = 0; k < num_cols; k++) {
            int grad_root = out3*i + out2*j+out1*k;
            int a_root = three_d_size*i + two_d_size*j+one_d_size * k;
            for( int m = 0 ; m < depth; m++) {
              int a_index = a_root+m;
              for (int l =0; l < num_offsets; l++ ) {
                int j_offset = offset_list[2*l];
                int k_offset = offset_list[2*l+1];
                int min_j = 0;
                int max_j = num_rows;
                int min_k = 0;
                int max_k = num_cols;
                if(j_offset < 0){
                    min_j = -1*j_offset;
                }else{
                    max_j -= j_offset;
                }
                if(k_offset < 0){
                    min_k = -1*k_offset;
                }else{
                    max_k -= k_offset;
                }

                int grad_index =  grad_root+ l;
                if( j >= min_j && j < max_j  && k >= min_k && k < max_k)
                {
                    int b_j = j+j_offset;
                    int b_k = k+k_offset;
                    int b_index = b_root + two_d_size*b_j + one_d_size * b_k +m;
                    float current_coefficient = grad[grad_index]/ depth;
                    out_a[a_index]+= current_coefficient*b[b_index];
                    out_b[b_index]+= current_coefficient*a[a_index];
	    
                }
              }
             }
           }
        }
      }


}


void CorrelationGradKernelLauncher(const float* a, const float*b, const float*grad, float* out_a,float*out_b, const int batch_size,const int num_rows, const int num_cols, const int depth,const int num_offsets, const int* offset_list) {
  int *offset_array;
  cudaMalloc(&offset_array, num_offsets * sizeof(int)); 
  cudaMemcpy(offset_array, offset_list, num_offsets*sizeof(int), cudaMemcpyHostToDevice);
  CorrelationGradKernel<<<32, 256>>>(a, b, grad, out_a,out_b,batch_size,num_rows,num_cols,depth,num_offsets,offset_array);
}

#endif
