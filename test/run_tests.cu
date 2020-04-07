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
/********************************************************************************************************************/

#include "../include/convolution.h"
#include "../include/max_pool.h"
#include "../include/relu.h"
#include "../include/softmax.h"
#include "../include/sigmoid.h"

/* Utility Functions */
void pprint(float* matrix, int size, int width){
    for(int i=0; i<size; i++){
        if(i%width==0) std::cout << std::endl;
        std::cout << matrix[i] << " ";
    }
    std::cout << std::endl;
}
/*********************/

void test_convolution_forward() {
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

    const float data_[5][4] = {{1, 6, 11, 16},
                               {2, 7, 12, 17},
                               {3, 8, 13, 18},
                               {4, 9, 14, 19},
                               {5, 10, 15, 20}};

    for(int i = 0;i < HEIGHT;i++)
      for(int j = 0;j < WIDTH;j++)
        cpu_data[i*WIDTH + j] = data_[i][j];

    // for(int i = 0;i < HEIGHT;i++) {
    //   for(int j = 0;j < WIDTH;j++)
        // std::cout << cpu_data[i*WIDTH + j] << " ";
      // std::cout << std::endl;
    // }

    checkCudaErrors(cudaMemcpyAsync(data, cpu_data, sizeof(float) * c.input_size,  cudaMemcpyHostToDevice));
    
    c.forward(data, output);

    // Move from device to host
    float *out = (float *)malloc(sizeof(float) * c.output_size);
    // float out[BATCH_SIZE][c.out_height][c.out_width][c.out_channels];
    checkCudaErrors(cudaMemcpy(out, output, sizeof(float) * c.output_size, cudaMemcpyDeviceToHost));

    for(int i = 0;i < c.output_size;i++) {
        // std::cout << out[i] << " ";
    }
    // std::cout << std::endl;
    // std::cout << c.output_size << std::endl;
}

void test_convolution_backward() {
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
         BATCH_SIZE, WIDTH, HEIGHT, true, true, GPU_ID, d1, d2, true);
    cudaMalloc(&data, sizeof(float) * c.input_size);
    cudaMalloc(&output, sizeof(float) * c.output_size);

    float *cpu_data = (float *)malloc(sizeof(float) * c.input_size);

    const float data_[5][4] = {{1, 6, 11, 16},
                               {2, 7, 12, 17},
                               {3, 8, 13, 18},
                               {4, 9, 14, 19},
                               {5, 10, 15, 20}};

    for(int i = 0;i < HEIGHT;i++)
      for(int j = 0;j < WIDTH;j++)
        cpu_data[i*WIDTH + j] = data_[i][j];

    // for(int i = 0;i < HEIGHT;i++) {
      // for(int j = 0;j < WIDTH;j++)
        // std::cout << cpu_data[i*WIDTH + j] << " ";
      // std::cout << std::endl;
    // }

    checkCudaErrors(cudaMemcpyAsync(data, cpu_data, sizeof(float) * c.input_size,  cudaMemcpyHostToDevice));
  
    c.forward(data, output);

    // Move from device to host
    float *out = (float *)malloc(sizeof(float) * c.output_size);
    checkCudaErrors(cudaMemcpy(out, output, sizeof(float) * c.output_size, cudaMemcpyDeviceToHost));

    c.backward(output, c.input_descriptor, data);

    int t = c.in_channels * c.kernel_size * c.kernel_size * c.out_channels;
    float *grad_kernel = (float *)malloc(sizeof(float) * t);
    checkCudaErrors(cudaMemcpy(grad_kernel, c.grad_kernel, sizeof(float) * t, cudaMemcpyDeviceToHost));

    // for(int i = 0;i < t;i++)
      // std::cout << grad_kernel[i] << " ";
    // std::cout << std::endl;

    t = c.out_channels;
    float *grad_bias = (float *)malloc(sizeof(float) * t);
    checkCudaErrors(cudaMemcpy(grad_bias, c.grad_bias, sizeof(float) * t, cudaMemcpyDeviceToHost));

    // for(int i = 0;i < t;i++)
      // std::cout << grad_bias[i] << " ";
    // std::cout << std::endl;

    t = BATCH_SIZE * HEIGHT * WIDTH * CHANNELS;
    float *grad_data = (float *)malloc(sizeof(float) * t);
    checkCudaErrors(cudaMemcpy(grad_data, c.grad_data, sizeof(float) * t, cudaMemcpyDeviceToHost));
    
    // for(int i = 0;i < t;i++)
      // std::cout << grad_data[i] << " ";
    // std::cout << std::endl;
}

void test_mpl(){
    // Initialize image and cudnn handles
    // std::cout << "-------- TESTING MAX POOL LAYER --------\n";
    int WIDTH_CONV = 4, HEIGHT_CONV = 5, KERNEL_SIZE_CONV=2, PADDING_CONV=1, STRIDE_CONV=1;  //Input to Conv
    int SIZE_MAX_POOL=2, STRIDE_MAX_POOL=2, PADDING_MAX_POOL=0, HEIGHT_MAX_POOL=(HEIGHT_CONV - KERNEL_SIZE_CONV + 2*PADDING_CONV)/STRIDE_CONV + 1, WIDTH_MAX_POOL=(WIDTH_CONV - KERNEL_SIZE_CONV + 2*PADDING_CONV)/STRIDE_CONV + 1;  //For MaxPool
    int BATCH_SIZE = 1, CHANNELS = 1;     //Image
    int GPU_ID = 0;
    checkCudaErrors(cudaSetDevice(GPU_ID));
    float *data, *output_conv, *output_max_pool, *input_diff_grad, *output_diff_grad;
    cudnnHandle_t cudnn;
    cublasHandle_t cublas;
    cudnnTensorDescriptor_t d1, d2; // dummy descriptors
    cudnnCreate(&cudnn);
    cublasCreate(&cublas);

    // Stack Layers
    Conv c(1, CHANNELS, KERNEL_SIZE_CONV, PADDING_CONV, STRIDE_CONV, cudnn,  cublas, BATCH_SIZE, WIDTH_CONV, HEIGHT_CONV, true, false, GPU_ID, d1, d2, true);
    MaxPoolLayer mpl(SIZE_MAX_POOL, STRIDE_MAX_POOL, PADDING_MAX_POOL, BATCH_SIZE, CHANNELS, HEIGHT_MAX_POOL, WIDTH_MAX_POOL, GPU_ID, cudnn);    

    //Initialize tensors device
    cudaMalloc(&data, sizeof(float) * c.input_size);                 //CONV INPUT
    cudaMalloc(&output_conv, sizeof(float) * c.output_size);         //CONV OUTPUT
    cudaMalloc(&output_max_pool, sizeof(float) * mpl.output_size);
    //Initalize arrays host
    float *cpu_data = (float *)malloc(sizeof(float) * c.input_size);
    for(int i = 0;i < c.input_size;i++) cpu_data[i] = i;
    checkCudaErrors(cudaMemcpyAsync(data, cpu_data, sizeof(float) * c.input_size,  cudaMemcpyHostToDevice));
    float* output_matrix = (float *)malloc(sizeof(float)*mpl.output_size);
    float* output_matrix_conv = (float *)malloc(sizeof(float)*mpl.input_size);

    // std::cout << "Input Matrix:";
    // pprint(cpu_data, c.input_size, WIDTH_CONV);
    // std::cout << "\nApply Convolution kernel_size=2, padding=1, stride=1:\n";
    c.forward(data, output_conv);
    checkCudaErrors(cudaMemcpy(output_matrix_conv, output_conv, sizeof(float)*mpl.input_size, cudaMemcpyDeviceToHost));
    // std::cout << "\nOutput Matrix From Convolution:";
    // pprint(output_matrix_conv, mpl.input_size, mpl.input_width);
    // std::cout << "\nPerforming max pool size=(2,2), stride=(2, 2), padding=(0, 0)\n";
    mpl.forward(output_conv, output_max_pool);
    checkCudaErrors(cudaMemcpy(output_matrix, output_max_pool, sizeof(float)*mpl.output_size, cudaMemcpyDeviceToHost));
    // std::cout << "\nOutput Matrix From Max Pool:";
    // pprint(output_matrix, mpl.output_size, mpl.output_width);
    
    //Generate a input differential gradient recieved by max pool layer in backprop
    cudaMalloc(&input_diff_grad, sizeof(float) * mpl.output_size);
    cudaMalloc(&output_diff_grad, sizeof(float) * mpl.input_size);
    float *input_diff_grad_cpu = (float *)malloc(sizeof(float) * mpl.output_size);
    for(int i = 0;i < mpl.output_size;i++) input_diff_grad_cpu[i] = 10.0;
    checkCudaErrors(cudaMemcpyAsync(input_diff_grad, input_diff_grad_cpu, sizeof(float) * mpl.output_size,  cudaMemcpyHostToDevice));
    float* output_gradient = (float *)malloc(sizeof(float)*mpl.input_size);
    
    mpl.backward(output_conv, input_diff_grad, output_max_pool, output_diff_grad);
    checkCudaErrors(cudaMemcpy(output_gradient, output_diff_grad, sizeof(float)*mpl.input_size, cudaMemcpyDeviceToHost));
    // std::cout << "\nGradient from Max Pool Layer:";
    // pprint(output_gradient, mpl.input_size, mpl.input_width);
    // printf("\n\n\nDone\n\n\n");
    // checkCudaErrors(cudaSetDevice(GPU_ID));
    return;
}

void test_relu() 
{
    int WIDTH = 5, HEIGHT = 5, BATCH_SIZE = 1, CHANNELS = 1;
    int GPU_ID = 0;
    checkCudaErrors(cudaSetDevice(GPU_ID));
 
    float *data, *output, *dup, *dout;
    cudnnHandle_t cudnn;
    cublasHandle_t cublas;

    //cudnnTensorDescriptor_t d1, d2; // dummy descriptors
    cudnnCreate(&cudnn);
    cublasCreate(&cublas);
 
    Relu R(CHANNELS, CHANNELS, cudnn, cublas, BATCH_SIZE, HEIGHT, WIDTH, GPU_ID);
 
    cudaMalloc((void **)&data, sizeof(float) * R.input_size);
    cudaMalloc((void **)&output, sizeof(float) * R.output_size);
    cudaMalloc((void **)&dout, sizeof(float) * R.output_size);
    cudaMalloc((void **)&dup, sizeof(float) * R.output_size);

    float *cpu_data = (float *)malloc(sizeof(float) * R.input_size);
    for(int i = 0; i < R.input_size; i++)
        cpu_data[i] = -12.0 + i;
    cpu_data[1] = 3234.0; //to check clipping
    cpu_data[20] = 3566.0;
    
    // std::cout<<"Testing Forward . . ."<<std::endl;
 
    // std::cout << "Input Matrix:"<<std::endl;
    // for(int i=0; i<R.input_size; i++)
    // {
    //     if(i%WIDTH==0)
    //         std::cout << "\n";
    //     std::cout << cpu_data[i] << " ";
    // }
    // std::cout << "\nApply ReLU:"<<std::endl;
 
    checkCudaErrors(cudaMemcpy(data, cpu_data, sizeof(float) * R.input_size,  cudaMemcpyHostToDevice));
    // std::cout << "\nApply ReLU 2:"<<std::endl;
    R.forward(data, output);
 
    float *out = (float *)malloc(sizeof(float) * R.output_size);
    checkCudaErrors(cudaMemcpy(out, output, sizeof(float) * R.output_size, cudaMemcpyDeviceToHost));
    // std::cout << "Output Matrix:"<<std::endl;
    // for(int i=0; i<R.output_size; i++)
    // {
    //     if(i%WIDTH==0)
    //         std::cout << "\n";
    //     std::cout << out[i] << " ";
    // }
    // std::cout<<std::endl;
    
 
    // std::cout<<"Testing Backward . . ."<<std::endl;
 
    float *cpu_dup = (float *)malloc(sizeof(float) * R.output_size);
    for(int i=0; i<R.output_size; i++)
        cpu_dup[i] = 100 + i;
 
    // std::cout << "Upstream Derivatives:";
    // for(int i=0; i<R.output_size; i++)
    // {
    //     if(i%WIDTH==0)
    //         std::cout << "\n";
    //     std::cout << cpu_dup[i] << " ";
    // }
    // std::cout<<std::endl;
 
    checkCudaErrors(cudaMemcpy(dup, cpu_dup, sizeof(float) * R.output_size,  cudaMemcpyHostToDevice));

    // std::cout << "\nApply Backward:"<<std::endl;
    R.backward(dup, dout);
 
    float *cpu_dout = (float *)malloc(sizeof(float) * R.input_size);
    checkCudaErrors(cudaMemcpy(cpu_dout, dout, sizeof(float) * R.input_size, cudaMemcpyDeviceToHost));
    // std::cout << "Back prop results :"<<std::endl;
    // for(int i=0; i<R.input_size; i++)
    // {
    //     if(i%WIDTH==0)
    //         std::cout << "\n";
    //     std::cout << cpu_dout[i] << " ";
    // }
    // std::cout<<std::endl;
}

void test_sigmoid() 
{
    int WIDTH = 5, HEIGHT = 5, BATCH_SIZE = 1, CHANNELS = 1;
    int GPU_ID = 0;
    checkCudaErrors(cudaSetDevice(GPU_ID));
 
    float *data, *output, *dup, *dout;
    cudnnHandle_t cudnn;
    cublasHandle_t cublas;

    //cudnnTensorDescriptor_t d1, d2; // dummy descriptors
    cudnnCreate(&cudnn);
    cublasCreate(&cublas);
 
    Sigmoid R(CHANNELS, CHANNELS, cudnn, cublas, BATCH_SIZE, HEIGHT, WIDTH, GPU_ID);
 
    cudaMalloc((void **)&data, sizeof(float) * R.input_size);
    cudaMalloc((void **)&output, sizeof(float) * R.output_size);
    cudaMalloc((void **)&dout, sizeof(float) * R.output_size);
    cudaMalloc((void **)&dup, sizeof(float) * R.output_size);

    float *cpu_data = (float *)malloc(sizeof(float) * R.input_size);
    for(int i = 0; i < R.input_size; i++)
        cpu_data[i] = -12.0 + i;
    cpu_data[1] = 3234.0; //to check clipping
    cpu_data[20] = 3566.0;
    
    // std::cout<<"Testing Forward . . ."<<std::endl;
 
    // std::cout << "Input Matrix:"<<std::endl;
    // for(int i=0; i<R.input_size; i++)
    // {
    //     if(i%WIDTH==0)
    //         std::cout << "\n";
    //     std::cout << cpu_data[i] << " ";
    // }
    // std::cout << "\nApply ReLU:"<<std::endl;
 
    checkCudaErrors(cudaMemcpy(data, cpu_data, sizeof(float) * R.input_size,  cudaMemcpyHostToDevice));
    // std::cout << "\nApply ReLU 2:"<<std::endl;
    R.forward(data, output);
 
    float *out = (float *)malloc(sizeof(float) * R.output_size);
    checkCudaErrors(cudaMemcpy(out, output, sizeof(float) * R.output_size, cudaMemcpyDeviceToHost));
    // std::cout << "Output Matrix:"<<std::endl;
    // for(int i=0; i<R.output_size; i++)
    // {
    //     if(i%WIDTH==0)
    //         std::cout << "\n";
    //     std::cout << out[i] << " ";
    // }
    // std::cout<<std::endl;
    
 
    // std::cout<<"Testing Backward . . ."<<std::endl;
 
    float *cpu_dup = (float *)malloc(sizeof(float) * R.output_size);
    for(int i=0; i<R.output_size; i++)
        cpu_dup[i] = 100 + i;
 
    // std::cout << "Upstream Derivatives:";
    // for(int i=0; i<R.output_size; i++)
    // {
    //     if(i%WIDTH==0)
    //         std::cout << "\n";
    //     std::cout << cpu_dup[i] << " ";
    // }
    // std::cout<<std::endl;
 
    checkCudaErrors(cudaMemcpy(dup, cpu_dup, sizeof(float) * R.output_size,  cudaMemcpyHostToDevice));

    // std::cout << "\nApply Backward:"<<std::endl;
    R.backward(dup, dout);
 
    float *cpu_dout = (float *)malloc(sizeof(float) * R.input_size);
    checkCudaErrors(cudaMemcpy(cpu_dout, dout, sizeof(float) * R.input_size, cudaMemcpyDeviceToHost));
    // std::cout << "Back prop results :"<<std::endl;
    // for(int i=0; i<R.input_size; i++)
    // {
    //     if(i%WIDTH==0)
    //         std::cout << "\n";
    //     std::cout << cpu_dout[i] << " ";
    // }
    // std::cout<<std::endl;
}

void test() {
    // Tests both backward and forward
    test_convolution_backward();
    printf("\n\n\n----------Convolution Test Passed!-----------\n\n\n");
    test_mpl();
    printf("\n\n\n----------Max-Pooling Test Passed!-----------\n\n\n");
    test_relu();
    printf("\n\n\n----------Relu Test Passed!-----------\n\n\n");
    test_sigmoid();
    printf("\n\n\n----------Sigmoid Test Passed!-----------\n\n\n");
}

int main() {
    test();
    return 0;
}
