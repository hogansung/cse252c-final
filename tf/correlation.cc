/// \file correlation.cc
/// \author Jgorgen
/// \brief Implementation of a pixel-wise correlation
/// operation in Tensorflow.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("Correlation")
  .Input("a: float")
  .Input("b: float")
  .Output("correlation: float")
  .Attr("stride: int = 2")
  .Attr("max_displacement: int = 20")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* context) {
    shape_inference::ShapeHandle input_shape = context->input(0);
    shape_inference::ShapeHandle output_shape;
    int max_displacement;
    int stride;
    context->GetAttr("stride", &stride);
    context->GetAttr("max_displacement", &max_displacement);
    TF_RETURN_IF_ERROR(context->WithRank(context->input(0), 4, &input_shape));
    TF_RETURN_IF_ERROR(context->WithRank(context->input(1), 4, &input_shape));
    int num_steps = 2*(max_displacement/stride) + 1;
    int num_outputs = num_steps*num_steps;
    context->ReplaceDim(input_shape, 3, context->MakeDim(shape_inference::DimensionOrConstant(num_outputs)), &output_shape);
    

    context->set_output(0, output_shape);
    return Status::OK();
  });

/// \brief Implementation of a correlation operation.
/// \param context
/// \author Jgorgen
class CorrelationOp : public OpKernel {
public:
  /// \brief Constructor.
  /// \param context
  explicit CorrelationOp(OpKernelConstruction* context) : OpKernel(context) {
        // Get the stride to
    OP_REQUIRES_OK(context,
                   context->GetAttr("stride", &stride_));
    // Check that stride is positive
    OP_REQUIRES(context, stride_ > 0,
                errors::InvalidArgument("Need stride > 0, got ",
                                        stride_));
        // Get the index of the max_displacement to preserve
    OP_REQUIRES_OK(context,
                   context->GetAttr("max_displacement", &max_displacement_));
    // Check that max_displacement is positive
    OP_REQUIRES(context, max_displacement_ > 0,
                errors::InvalidArgument("Need max_displacement > 0, got ",
                                        max_displacement_));
  }
  int stride_;
  int max_displacement_;
  
  /// \brief Compute the inner product.
  /// \param context
  void Compute(OpKernelContext* context) override {
    
    // some checks to be sure ...
    DCHECK_EQ(2, context->num_inputs());
    
    // get the left tensor
    const Tensor& a = context->input(0);
    
    // get the right tensor
    const Tensor& b = context->input(1);
    
    // check shapes of input and weights
    const TensorShape& a_shape = a.shape();
    const TensorShape& b_shape = b.shape();
    
    // check inputs are both (batch_size,height,width,num_channels)
    DCHECK_EQ(a_shape.dims(), 4);
    DCHECK_EQ(b_shape.dims(), 4);
    DCHECK_EQ(a_shape.dim_size(0), b_shape.dim_size(0));
    DCHECK_EQ(a_shape.dim_size(1), b_shape.dim_size(1));
    DCHECK_EQ(a_shape.dim_size(2), b_shape.dim_size(2));
    DCHECK_EQ(a_shape.dim_size(3), b_shape.dim_size(3));
                


    // create output shape
    TensorShape output_shape;
    int num_steps = 2*(max_displacement_/stride_) + 1;
    int num_outputs = num_steps*num_steps;

    output_shape.AddDim(a_shape.dim_size(0));
    output_shape.AddDim(a_shape.dim_size(1));
    output_shape.AddDim(a_shape.dim_size(2));
    output_shape.AddDim(num_outputs);
    std::vector<std::pair<int,int> > offsets(num_outputs);
    size_t offset_index = 0;
    for(int j = -this->max_displacement_; j<= this->max_displacement_;  j+= this->stride_)
    {
        for(int k= -this->max_displacement_; k <= this->max_displacement_; k+= this->stride_)
        {
            offsets.at(offset_index).first = j;
            offsets.at(offset_index).second = k;
            offset_index++;
        }
    }
            
    // create output tensor
    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    
    // get the corresponding Eigen tensors for data access
    auto a_tensor = a.tensor<float,4>();
    auto b_tensor = b.tensor<float,4>();
    auto output_tensor = output->tensor<float,4>();
    
    for (int i = 0; i < output->shape().dim_size(0); i++) {
      for (int l = 0; l < output->shape().dim_size(3); l++) {
        int j_offset = offsets[l].first;
        int k_offset = offsets[l].second;
        int min_j = 0;
        int max_j = output->shape().dim_size(1);
        int min_k = 0;
        int max_k = output->shape().dim_size(2);
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
        // Zero out the top few rows
        for (int j= 0; j<min_j; j++)
        {
            for(int k=0; k<output->shape().dim_size(2);k++)
            {
               output_tensor(i, j,k,l) =0 ;
            }
        }
        // Zero out the bottom few rows
        for (int j= max_j; j<output->shape().dim_size(1); j++)
        {
            for(int k=0; k<output->shape().dim_size(2);k++)
            {
               output_tensor(i, j,k,l) =0 ;
            }
        }

        //Zero out the left and right few columns

        for (int j= min_j; j<max_j; j++)
        {
            for(int k=0; k<min_k;k++)
            {
               output_tensor(i, j,k,l) =0 ;
            }
            for(int k=max_k; k<output->shape().dim_size(2);k++)
            {
               output_tensor(i, j,k,l) =0 ;
            }
        }



        // Fill the rest with the dot product of a(i,j,k) and b(i,j+j_offset,k+k_offset)
        for (int j = min_j; j < max_j; j++) {
          for (int k = min_k; k < max_k; k++) {
                 output_tensor(i, j,k,l) =0 ;
                 int max_m = a.shape().dim_size(3);
                 for( int m = 0 ; m < a.shape().dim_size(3); m++) {
                     output_tensor(i,j,k,l)+= a_tensor(i,j,k,m)*b_tensor(i,j+j_offset,k+k_offset,m);
                  }
                  output_tensor(i,j,k,l)/= max_m;
            }

        }
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("Correlation").Device(DEVICE_CPU), CorrelationOp);
