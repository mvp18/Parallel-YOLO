#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cfloat>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>
#include <cudnn.h>

/*** Definitions ***/
// Block width for CUDA kernels
#define BW 128
#define RANDOM_SEED -1

#ifdef USE_GFLAGS
    #include <gflags/gflags.h>

    #ifndef _WIN32
        #define gflags google
    #endif
#else
    // Constant versions of gflags
    #define DEFINE_int32(flag, default_value, description) const int FLAGS_##flag = (default_value)
    #define DEFINE_uint64(flag, default_value, description) const unsigned long long FLAGS_##flag = (default_value)
    #define DEFINE_bool(flag, default_value, description) const bool FLAGS_##flag = (default_value)
    #define DEFINE_double(flag, default_value, description) const double FLAGS_##flag = (default_value)
    #define DEFINE_string(flag, default_value, description) const std::string FLAGS_##flag ((default_value))
#endif

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(1);                                                           \
} while(0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCudaErrors(status) do {                                   \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#include "layers.h"

void test_forward_mpl();
void test_forward_conv();
void test_mpl();
void test_conv();
void pprint(float* , int, int);

int main() {
    std::cout << "---------------JUST RUNNING CONVOLUTION---------------\n";
    test_forward_conv();
    std::cout << "\n---------------JUST RUNNING MAX POOL---------------\n";
    test_forward_mpl();
    std::cout << "\n---------------RUNNING CONV THEN MAX POOL---------------\n";

    // Initialize image and cudnn handles
    int WIDTH_CONV = 4, HEIGHT_CONV = 5, KERNEL_SIZE_CONV=2, PADDING_CONV=1, STRIDE_CONV=1;  //Input to Conv
    int SIZE_MAX_POOL=2, STRIDE_MAX_POOL=2, PADDING_MAX_POOL=0, HEIGHT_MAX_POOL=(HEIGHT_CONV - KERNEL_SIZE_CONV + 2*PADDING_CONV)/STRIDE_CONV + 1, WIDTH_MAX_POOL=(WIDTH_CONV - KERNEL_SIZE_CONV + 2*PADDING_CONV)/STRIDE_CONV + 1;  //For MaxPool
    int BATCH_SIZE = 1, CHANNELS = 1;     //Image
    int GPU_ID = 0;
    checkCudaErrors(cudaSetDevice(GPU_ID));
    float *data, *output_conv, *output_max_pool;
    cudnnHandle_t cudnn;
    cublasHandle_t cublas;
    cudnnTensorDescriptor_t d1, d2; // dummy descriptors
    cudnnCreate(&cudnn);
    cublasCreate(&cublas);

    // Stack Layers
    Conv c(1, CHANNELS, KERNEL_SIZE_CONV, PADDING_CONV, STRIDE_CONV, cudnn,  cublas, BATCH_SIZE, WIDTH_CONV, HEIGHT_CONV, true, false, GPU_ID, d1, d2, true);
    MaxPoolLayer mpl(SIZE_MAX_POOL, STRIDE_MAX_POOL, PADDING_MAX_POOL, BATCH_SIZE, CHANNELS, HEIGHT_MAX_POOL, WIDTH_MAX_POOL, GPU_ID, cudnn);    

    //Initialize tensors device
    cudaMalloc(&data, sizeof(float) * c.input_size);            //CONV INPUT
    cudaMalloc(&output_conv, sizeof(float) * c.output_size);    //CONV OUTPUT
    cudaMalloc(&output_max_pool, sizeof(float) * mpl.output_size);
    //Initalize arrays host
    float *cpu_data = (float *)malloc(sizeof(float) * c.input_size);
    for(int i = 0;i < c.input_size;i++) cpu_data[i] = 1.0;
    checkCudaErrors(cudaMemcpyAsync(data, cpu_data, sizeof(float) * c.input_size,  cudaMemcpyHostToDevice));
    float* output_matrix = (float *)malloc(sizeof(float)*mpl.output_size);
    
    std::cout << "Input Matrix:";
    pprint(cpu_data, c.input_size, WIDTH_CONV);
    std::cout << "\nApply Convolution kernel_size=2, padding=1, stride=1:\n";
    c.forward(data, output_conv);
    std::cout << "Performing max pool size=(2,2), stride=(2, 2), padding=(0, 0)\n";
    mpl.forward(output_conv, output_max_pool);
    checkCudaErrors(cudaMemcpy(output_matrix, output_max_pool, sizeof(float)*mpl.output_size, cudaMemcpyDeviceToHost));
    std::cout << "\nOutput Matrix:";
    pprint(output_matrix, mpl.output_size, mpl.output_width);
    return 0;
}

void test_forward_mpl(){
  // Take 5x5 image, use 3x3 stride
    int WIDTH = 4, HEIGHT = 4, BATCH_SIZE = 1, CHANNELS = 1, SIZE=2, STRIDE=2, PADDING=0;
    float *data, *output;
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    MaxPoolLayer mpl(SIZE, STRIDE, PADDING, BATCH_SIZE, CHANNELS, HEIGHT, WIDTH, 0, cudnn);
    
    float* input_matrix = (float *)malloc(sizeof(float)*mpl.input_size);
    float* output_matrix = (float *)malloc(sizeof(float)*mpl.output_size);
    for(int i=0; i<mpl.input_size; i++) input_matrix[i]=i;
    cudaMalloc(&data, sizeof(float) * mpl.input_size);
    cudaMalloc(&output, sizeof(float) * mpl.output_size);
    checkCudaErrors(cudaMemcpyAsync(data, input_matrix, sizeof(float)*mpl.input_size, cudaMemcpyHostToDevice));
    
    std::cout << "Input Matrix:";
    for(int i=0; i<mpl.input_size; i++){
    if(i%WIDTH==0) std::cout << "\n";
    std::cout << input_matrix[i] << " ";
  }
  
  std::cout << "\n\nPerforming max pool size=(2,2), stride=(2, 2), padding=(0, 0)\n";
  mpl.forward(data, output);
  
  checkCudaErrors(cudaMemcpy(output_matrix, output, sizeof(float)*mpl.output_size, cudaMemcpyDeviceToHost));
  std::cout << "\nOutput Matrix:";
  for(int i=0; i<mpl.output_size; i++){
    if(i%mpl.output_width==0) std::cout << "\n";
    std::cout << output_matrix[i] << " ";
  }
  std::cout << "\n";
}


void test_forward_conv() {
    int WIDTH = 4, HEIGHT = 5, BATCH_SIZE = 1, CHANNELS = 1;
    int GPU_ID = 0;
    checkCudaErrors(cudaSetDevice(GPU_ID));
    float *data, *output;
    cudnnHandle_t cudnn;
    cublasHandle_t cublas;

    cudnnTensorDescriptor_t d1, d2; // dummy descriptors
    cudnnCreate(&cudnn);
    cublasCreate(&cublas);
    Conv c(1, CHANNELS, 2, 1, 1, cudnn, cublas,
         BATCH_SIZE, WIDTH, HEIGHT, true, false, GPU_ID, d1, d2, true);
    cudaMalloc(&data, sizeof(float) * c.input_size);
    cudaMalloc(&output, sizeof(float) * c.output_size);

    float *cpu_data = (float *)malloc(sizeof(float) * c.input_size);
    for(int i = 0;i < c.input_size;i++) cpu_data[i] = 1.0;
    checkCudaErrors(cudaMemcpyAsync(data, cpu_data, sizeof(float) * c.input_size,  cudaMemcpyHostToDevice));
    std::cout << "Input Matrix:";
    for(int i=0; i<c.input_size; i++){
        if(i%WIDTH==0) std::cout << "\n";
        std::cout << cpu_data[i] << " ";
    }
    std::cout << "\nApply Convolution:";
    c.forward(data, output);

    // Move from device to host
    float *out = (float *)malloc(sizeof(float) * c.output_size);
    // float out[BATCH_SIZE][c.out_height][c.out_width][c.out_channels];
    checkCudaErrors(cudaMemcpy(out, output, sizeof(float) * c.output_size, cudaMemcpyDeviceToHost));
    for(int i = 0;i < c.output_size;i++) {
        if(i%(((WIDTH - 2 + 2*1)/1) + 1)==0) std::cout << "\n";
        std::cout << out[i] << " ";
    }
    std::cout << std::endl;
}

void test_conv() {
    test_forward_conv();
}

void test_mpl() {
    test_forward_mpl();
}

void pprint(float* matrix, int size, int width){
    for(int i=0; i<size; i++){
        if(i%width==0) std::cout << std::endl;
        std::cout << matrix[i] << " ";
    }
    std::cout << std::endl;
}
