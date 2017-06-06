/// \file correlation_grad.cc
/// \author David Stutz
/// \brief Implementation of the gradient of a inner product operation, see
/// correlation.cc.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

// the gradients are simply passed as additional arguments as
// they are available in the Python function for registering the gradient operation.
REGISTER_OP("CorrelationGrad")
  .Input("grad: float32")
  .Input("a: float32")
  .Input("b: float32")
  .Output("grad_a: float32")
  .Output("grad_b: float32")
  .Attr("stride: int = 2")
  .Attr("max_displacement: int = 20");

/// \brief Implementation of an inner product gradient operation.
/// Note that this operation is used in Python to register the gradient as
/// this is not possible in C*+ right now.
/// \param context
/// \author David Stutz
class CorrelationGradOp : public OpKernel {
public:
  /// \brief Constructor.
  /// \param context
  explicit CorrelationGradOp(OpKernelConstruction* context) : OpKernel(context) {
            // Get the stride to
    OP_REQUIRES_OK(context,
                   context->GetAttr("stride", &stride_));
    // Check that stride is positive
    OP_REQUIRES(context, stride_ > 0,
                errors::InvalidArgument("Need stride > 0, got ",
                                        stride_));
        // Get the index of the max_displacement
    OP_REQUIRES_OK(context,
                   context->GetAttr("max_displacement", &max_displacement_));
    // Check that max_displacement is positive
    OP_REQUIRES(context, max_displacement_ > 0,
                errors::InvalidArgument("Need max_displacement > 0, got ",
                                        max_displacement_));
  }
  int stride_;
  int max_displacement_;
  
  /// \brief Compute the inner product gradients.
  /// \param context
  void Compute(OpKernelContext* context) override {
    
    // output and grad is provided as input
    DCHECK_EQ(3, context->num_inputs());

    // get the gradient tensor
    const Tensor& grad = context->input(0);
    
    // get the original input tensor
    const Tensor& a = context->input(1);
    
    // get the weight tensor
    const Tensor& b = context->input(2);
    
    // create input shape (inferred from the additional attribute `n`)
    TensorShape a_shape = a.shape();
    TensorShape b_shape = b.shape();
    
    DCHECK_EQ(a_shape.dim_size(0), b_shape.dim_size(0));
    DCHECK_EQ(a_shape.dim_size(1), b_shape.dim_size(1));
    DCHECK_EQ(a_shape.dim_size(2), b_shape.dim_size(2));
    DCHECK_EQ(a_shape.dim_size(3), b_shape.dim_size(3));
    DCHECK_EQ(b_shape.dim_size(0), grad.shape().dim_size(0));
    
    // create output tensors
    Tensor* grad_a = NULL;
    Tensor* grad_b = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, a_shape, &grad_a));
    OP_REQUIRES_OK(context, context->allocate_output(1, b_shape, &grad_b));
    
    // get the Eigen tensors for data access
    auto grad_tensor = grad.tensor<float,4>();
    auto a_tensor = a.tensor<float,4>();
    auto b_tensor = b.tensor<float,4>();
    auto grad_a_tensor = grad_a->tensor<float,4>();
    auto grad_b_tensor = grad_b->tensor<float,4>();

    int stride = stride_;
    int max_displacement = max_displacement_;



    int num_steps = 2*(max_displacement/stride) + 1;
    int num_outputs = num_steps*num_steps;
    DCHECK_EQ(grad.shape().dim_size(3),num_outputs);

    std::vector<std::pair<int,int> > offsets(num_outputs);
    size_t offset_index = 0;


    for(int j = -max_displacement; j<= max_displacement;  j+= stride)
    {
        for(int k= -max_displacement; k <= max_displacement; k+= stride)
        {
            offsets.at(offset_index).first = j;
            offsets.at(offset_index).second = k;
            offset_index++;
        }
    }
    
    // Zero out the outputs
    int max_m      =  a.shape().dim_size(3);
    int batch_size =  grad_a->shape().dim_size(0);
    int num_rows   =  grad_a->shape().dim_size(1);
    int num_cols   =  grad_a->shape().dim_size(2);
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_rows; j++) {
          for (int k = 0; k < num_cols; k++) {
                 for( int m = 0 ; m < max_m; m++) {
                     grad_a_tensor(i,j,k,m) = 0;
                     grad_b_tensor(i,j,k,m) = 0;
                 }
          }
        }
    }
        
    // Iterate over the input_gradient, mapping it to the output gradient where necessary
    for (int i = 0; i < batch_size; i++) {
        // Fill the rest with the dot product of a(i,j,k) and b(i,j+j_offset,k+k_offset)
        for (int j = 0; j < num_rows; j++) {
          for (int k = 0; k < num_cols; k++) {
            for( int m = 0 ; m < max_m; m++) {
              for (int l = 0; l < num_outputs; l++) {
                int j_offset = offsets[l].first;
                int k_offset = offsets[l].second;
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
                if( j >= min_j && j < max_j  && k >= min_k && k < max_k)
                {
                   float current_coefficient = grad_tensor(i,j,k,l)/ max_m;
                     grad_a_tensor(i,j,k,m)+= current_coefficient*b_tensor(i,j+j_offset,k+k_offset,m);
                     grad_b_tensor(i,j+j_offset,k+k_offset,m)+= current_coefficient*a_tensor(i,j,k,m);
                 }
               }
            }
        }
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("CorrelationGrad").Device(DEVICE_CPU), CorrelationGradOp);
