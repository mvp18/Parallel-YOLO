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

include_directories($ENV{CUDNN_PATH} $ENV{CUDNN_PATH}/include)
link_directories($ENV{CUDNN_PATH} $ENV{CUDNN_PATH}/lib $ENV{CUDNN_PATH}/lib64)





cuda_add_executable(conv layers/convolution.cu)
cuda_add_cublas_to_target(conv)

if(USE_GFLAGS)
  target_link_libraries(conv gflags cudnn)
else()
  target_link_libraries(conv cudnn)
endif()




#cuda_add_executable(FC layers/fully_connected.cu)
#cuda_add_cublas_to_target(FC)

#if(USE_GFLAGS)
#  target_link_libraries(FC gflags cudnn)
#else()
#  target_link_libraries(FC cudnn)
#endif()



cuda_add_executable(max_pool layers/max_pool.cu)
cuda_add_cublas_to_target(max_pool)

if(USE_GFLAGS)
  target_link_libraries(max_pool gflags cudnn)
else()
  target_link_libraries(max_pool cudnn)
endif()


#cuda_add_executable(relu layers/relu.cu)
#cuda_add_cublas_to_target(relu)

#if(USE_GFLAGS)
#  target_link_libraries(relu gflags cudnn)
#else()
#  target_link_libraries(relu cudnn)
#endif()



#cuda_add_executable(softmax layers/softmax.cu)
#cuda_add_cublas_to_target(softmax)

#if(USE_GFLAGS)
#  target_link_libraries(softmax gflags cudnn)
#else()
#  target_link_libraries(softmax cudnn)
#endif()




#cuda_add_executable(run combine_layers.cu layers.h)
#cuda_add_cublas_to_target(run)

#if(USE_GFLAGS)
#  target_link_libraries(run gflags cudnn)
#else()
#  target_link_libraries(run cudnn)
#endif()
