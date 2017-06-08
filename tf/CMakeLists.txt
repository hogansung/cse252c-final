cmake_minimum_required(VERSION 2.8)

# get tensorflow include dirs, see https://www.tensorflow.org/how_tos/adding_an_op/
execute_process(COMMAND python -c "import tensorflow; print(tensorflow.sysconfig.get_include())" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS)

# C++11 required for tensorflow
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

# build the actual operation which can be used directory
include_directories(${Tensorflow_INCLUDE_DIRS})
add_library(correlation SHARED correlation.cc correlation_gpu.cu.o)
target_link_libraries(correlation cudart)

# build the gradient operation which is used in correlation_grad.py
# to register it
include_directories(${Tensorflow_INCLUDE_DIRS})
#add_library(correlation_grad SHARED correlation_grad.cc)
add_library(correlation_grad SHARED correlation_grad.cc correlation_grad_gpu.cu.o)
target_link_libraries(correlation_grad cudart)
