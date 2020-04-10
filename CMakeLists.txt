cmake_minimum_required (VERSION 2.6)
project (cudnn-training)

find_package(CUDA 6.5 REQUIRED)

# Uncomment the following line to use gflags
#set(USE_GFLAGS 1)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("Debug mode")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_50,code=compute_50;-std=c++11;-g;-lineinfo;-Xcompiler;-ggdb)
else()
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_50,code=compute_50;-std=c++11;-O3;-DNDEBUG;-Xcompiler;-DNDEBUG)
endif()

set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# Addresses a bug where code is not compiled as C++11 in non-CUDA code and older g++ versions
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if(USE_GFLAGS)
  add_definitions(-DUSE_GFLAGS)
endif()
find_package(PNG REQUIRED)

include_directories($ENV{CUDNN_PATH} $ENV{CUDNN_PATH}/include)
include_directories(${PNG_INCLUDE_DIR})
link_directories($ENV{CUDNN_PATH} $ENV{CUDNN_PATH}/lib $ENV{CUDNN_PATH}/lib64)

cuda_add_executable(run_test test/run_tests.cu include/convolution.h include/max_pool.h include/relu.h include/softmax.h include/sigmoid.h)
cuda_add_executable(train src/model.cu include/convolution.h include/max_pool.h include/relu.h include/softmax.h include/sigmoid.h)

cuda_add_cublas_to_target(run_test)
cuda_add_cublas_to_target(train)

if(USE_GFLAGS)
  target_link_libraries(run_test gflags cudnn ${PNG_LIBRARY})
  target_link_libraries(train gflags cudnn ${PNG_LIBRARY})
else()
  target_link_libraries(run_test cudnn ${PNG_LIBRARY})
  target_link_libraries(train cudnn ${PNG_LIBRARY})
endif()
