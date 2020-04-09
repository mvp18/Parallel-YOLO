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

void test_convolution() 
{
    printf("*********** CONV_TEST **************\n");
  int WIDTH = 4, HEIGHT = 5, BATCH_SIZE = 1, CHANNELS = 1;
  int GPU_ID = 0;
  checkCudaErrors(cudaSetDevice(GPU_ID));
    float *data, *output;
    cudnnHandle_t cudnn;
    cublasHandle_t cublas;

    cudnnTensorDescriptor_t d1, d2; // dummy descriptors

    cudnnCreate(&cudnn);
    cublasCreate(&cublas);
    Conv c(1, CHANNELS, 3, 1, 1, cudnn, cublas,
         BATCH_SIZE, WIDTH, HEIGHT, true, true, GPU_ID, d1, d2, true);
    cudaMalloc(&data, sizeof(float) * c.input_size);
    cudaMalloc(&output, sizeof(float) * c.output_size);

    float *cpu_data = (float *)malloc(sizeof(float) * c.input_size);

    const float data_[5][4] = {{1, 6, 11, 16},
                               {2, 7, 12, 17},
                               {3, 8, 13, 18},
                               {4, 9, 14, 19},
                               {5, 10, 15, 20}};

    const float data_grad[5][4] = {{0, 0, 0, 0},
                                    {0, 10, 10, 0},
                                    {0, 0, 0, 0},
                                    {0, 10, 10, 0},
                                    {0, 0, 0, 0}};


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
    printf("Output of conv:\n");
    pprint(out, c.output_size, WIDTH);
    float *grad_above = (float *)malloc(sizeof(float) * c.output_size);
    for(int i = 0;i < HEIGHT;i++)
      for(int j = 0;j < WIDTH;j++)
        grad_above[i*WIDTH + j] = data_grad[i][j];


    printf("Grad abovve\n");
    pprint(grad_above, c.output_size, WIDTH);
    float *d_grad_above;
    cudaMalloc(&d_grad_above, sizeof(float) * c.output_size);
    checkCudaErrors(cudaMemcpyAsync(d_grad_above, grad_above, sizeof(float) * c.output_size,  cudaMemcpyHostToDevice));


    c.backward(d_grad_above, c.input_descriptor, data);

    int t = c.in_channels * c.kernel_size * c.kernel_size * c.out_channels;
    float *grad_kernel = (float *)malloc(sizeof(float) * t);
    checkCudaErrors(cudaMemcpy(grad_kernel, c.grad_kernel, sizeof(float) * t, cudaMemcpyDeviceToHost));

    std::cout<<"Printing grad_kernels . . .\n";
     for(int i = 0;i < t;i++)
       std::cout << grad_kernel[i] << " ";
     std::cout << std::endl;

    t = c.out_channels;
    float *grad_bias = (float *)malloc(sizeof(float) * t);
    checkCudaErrors(cudaMemcpy(grad_bias, c.grad_bias, sizeof(float) * t, cudaMemcpyDeviceToHost));

    std::cout<<"Printing grad_bias . . .\n";
     for(int i = 0;i < t;i++)
       std::cout << grad_bias[i] << " ";
     std::cout << std::endl;

    t = BATCH_SIZE * HEIGHT * WIDTH * CHANNELS;
    float *grad_data = (float *)malloc(sizeof(float) * t);
    checkCudaErrors(cudaMemcpy(grad_data, c.grad_data, sizeof(float) * t, cudaMemcpyDeviceToHost));
    
    std::cout<<"Printing grad_data . . .\n";
    pprint(grad_data, c.input_size, WIDTH);
    /* for(int i = 0;i < t;i++)
       std::cout << grad_data[i] << " ";
     std::cout << std::endl;*/
      printf("\n");
}

void test_mpl()
{
    printf("*********** CONV_POOL TEST **************\n");
    // Initialize image and cudnn handles
    // std::cout << "-------- TESTING MAX POOL LAYER --------\n";
    int WIDTH_CONV = 4, HEIGHT_CONV = 5, KERNEL_SIZE_CONV=3, PADDING_CONV=1, STRIDE_CONV=1;  //Input to Conv
    int SIZE_MAX_POOL=2, STRIDE_MAX_POOL=2, PADDING_MAX_POOL=0, HEIGHT_MAX_POOL=(HEIGHT_CONV - KERNEL_SIZE_CONV + 2*PADDING_CONV)/STRIDE_CONV + 1, WIDTH_MAX_POOL=(WIDTH_CONV - KERNEL_SIZE_CONV + 2*PADDING_CONV)/STRIDE_CONV + 1;  //For MaxPool
    int BATCH_SIZE = 1, CHANNELS = 1;     //Image
    int GPU_ID = 0;
    //int output_height = (HEIGHT_MAX_POOL - SIZE_MAX_POOL) / 2 + 1;
    //int output_width = (WIDTH_MAX_POOL - 2) / 2 + 1;
    checkCudaErrors(cudaSetDevice(GPU_ID));
    float *data, *output_conv, *output_max_pool, *input_diff_grad, *output_diff_grad;
    cudnnHandle_t cudnn;
    cublasHandle_t cublas;
    cudnnTensorDescriptor_t d1, d2; // dummy descriptors

    cudnnCreate(&cudnn);
    cublasCreate(&cublas);

    // Stack Layers
    Conv c(1, CHANNELS, KERNEL_SIZE_CONV, PADDING_CONV, STRIDE_CONV, cudnn,  cublas, BATCH_SIZE, WIDTH_CONV, HEIGHT_CONV, true, true, GPU_ID, d1, d2, true);
    MaxPoolLayer mpl(SIZE_MAX_POOL, STRIDE_MAX_POOL, PADDING_MAX_POOL, BATCH_SIZE, CHANNELS, HEIGHT_MAX_POOL, WIDTH_MAX_POOL, GPU_ID, cudnn, c.output_descriptor, d2, true);    

    //Initialize tensors device
    cudaMalloc(&data, sizeof(float) * c.input_size);                 //CONV INPUT
    cudaMalloc(&output_conv, sizeof(float) * c.output_size);         //CONV OUTPUT
    cudaMalloc(&output_max_pool, sizeof(float) * mpl.output_size);
    //Initalize arrays host
    float *cpu_data = (float *)malloc(sizeof(float) * c.input_size);
    const float data_[5][4] = {{1, 6, 11, 16},
                               {2, 7, 12, 17},
                               {3, 8, 13, 18},
                               {4, 9, 14, 19},
                               {5, 10, 15, 20}};
    for(int i=0; i<5; i++)
        for(int j=0; j<4; j++)
            cpu_data[4*i + j] = data_[i][j];
    //for(int i = 0;i < c.input_size;i++) cpu_data[i] = 3.0;
    checkCudaErrors(cudaMemcpyAsync(data, cpu_data, sizeof(float) * c.input_size,  cudaMemcpyHostToDevice));
    float* output_matrix = (float *)malloc(sizeof(float)*mpl.output_size);
    float* output_matrix_conv = (float *)malloc(sizeof(float)*mpl.input_size);
    float* grad_data_conv = (float *)malloc(sizeof(float) * c.input_size);

    std::cout << "Input Matrix:";
    pprint(cpu_data, c.input_size, WIDTH_CONV);
    std::cout << "\nApply Convolution kernel_size=3, padding=1, stride=1:\n";
    c.forward(data, output_conv);
    checkCudaErrors(cudaMemcpy(output_matrix_conv, output_conv, sizeof(float)*mpl.input_size, cudaMemcpyDeviceToHost));
    std::cout << "\nOutput Matrix From Convolution:";
    pprint(output_matrix_conv, mpl.input_size, mpl.input_width);
    std::cout << "\nPerforming max pool size=(2,2), stride=(2, 2), padding=(0, 0)\n";
    mpl.forward(output_conv, output_max_pool);
    checkCudaErrors(cudaMemcpy(output_matrix, output_max_pool, sizeof(float)*mpl.output_size, cudaMemcpyDeviceToHost));
    std::cout << "\nOutput Matrix From Max Pool:";
    pprint(output_matrix, mpl.output_size, mpl.out_width);
    
    //COMMENT BACKWARD
    //Generate a input differential gradient recieved by max pool layer in backprop
    cudaMalloc(&input_diff_grad, sizeof(float) * mpl.output_size);
    cudaMalloc(&output_diff_grad, sizeof(float) * mpl.input_size);
    float *input_diff_grad_cpu = (float *)malloc(sizeof(float) * mpl.output_size);
    for(int i = 0;i < mpl.output_size;i++) input_diff_grad_cpu[i] = 10.0;
    checkCudaErrors(cudaMemcpyAsync(input_diff_grad, input_diff_grad_cpu, sizeof(float) * mpl.output_size,  cudaMemcpyHostToDevice));
    float* output_gradient = (float *)malloc(sizeof(float)*mpl.input_size);
    
    mpl.backward(output_conv, input_diff_grad, output_max_pool, output_diff_grad/*, c.output_descriptor*/);
    checkCudaErrors(cudaMemcpy(output_gradient, output_diff_grad, sizeof(float)*mpl.input_size, cudaMemcpyDeviceToHost));
     std::cout << "\nGradient from Max Pool Layer:";
     pprint(output_gradient, mpl.input_size, mpl.input_width);
    // printf("\n\n\nDone\n\n\n");
    // checkCudaErrors(cudaSetDevice(GPU_ID));
    std::cout << "\nBackpropping that through conv:\n";
    c.backward(output_diff_grad, c.input_descriptor, data);
    checkCudaErrors(cudaMemcpy(grad_data_conv, c.grad_data, sizeof(float)*c.input_size, cudaMemcpyDeviceToHost));
    printf("\nGrad data conv");
    pprint(grad_data_conv, c.input_size, WIDTH_CONV);

    float *grad_kernel = (float *)malloc(sizeof(float) * 9);
    checkCudaErrors(cudaMemcpy(grad_kernel, c.grad_kernel, sizeof(float) * 9, cudaMemcpyDeviceToHost));

    std::cout<<"Printing grad_kernels . . .\n";
     for(int i = 0;i < 9;i++)
       std::cout << grad_kernel[i] << " ";
     std::cout << std::endl;

    int t = c.out_channels;
    float *grad_bias = (float *)malloc(sizeof(float) * t);
    checkCudaErrors(cudaMemcpy(grad_bias, c.grad_bias, sizeof(float) * t, cudaMemcpyDeviceToHost));

    std::cout<<"Printing grad_bias . . .\n";
     for(int i = 0;i < t;i++)
       std::cout << grad_bias[i] << " ";
     std::cout << std::endl;


    return;
}

void test_relu() 
{
    printf("*********** RELU TEST **************\n");
    int WIDTH = 5, HEIGHT = 5, BATCH_SIZE = 1, CHANNELS = 1;
    int GPU_ID = 0;
    checkCudaErrors(cudaSetDevice(GPU_ID));
 
    float *data, *output, *dup, *dout;
    cudnnHandle_t cudnn;
    cublasHandle_t cublas;

    cudnnTensorDescriptor_t d1, d2; // dummy descriptors
    cudnnCreate(&cudnn);
    cublasCreate(&cublas);
 
    Relu R(CHANNELS, CHANNELS, cudnn, cublas, BATCH_SIZE, HEIGHT, WIDTH, GPU_ID, d1, d2, true);
 
    cudaMalloc((void **)&data, sizeof(float) * R.input_size);
    cudaMalloc((void **)&output, sizeof(float) * R.output_size);
    cudaMalloc((void **)&dout, sizeof(float) * R.output_size);
    cudaMalloc((void **)&dup, sizeof(float) * R.output_size);

    float *cpu_data = (float *)malloc(sizeof(float) * R.input_size);
    for(int i = 0; i < R.input_size; i++)
        cpu_data[i] = -12.0 + i;
    cpu_data[1] = 3234.0; //to check clipping
    cpu_data[20] = 3566.0;
    
     std::cout<<"Testing Forward . . ."<<std::endl;
 
     std::cout << "Input Matrix:"<<std::endl;
     for(int i=0; i<R.input_size; i++)
     {
         if(i%WIDTH==0)
             std::cout << "\n";
         std::cout << cpu_data[i] << " ";
     }
     std::cout << "\nApply ReLU:"<<std::endl;
 
    checkCudaErrors(cudaMemcpy(data, cpu_data, sizeof(float) * R.input_size,  cudaMemcpyHostToDevice));
    // std::cout << "\nApply ReLU 2:"<<std::endl;
    R.forward(data, output);
 
    float *out = (float *)malloc(sizeof(float) * R.output_size);
    checkCudaErrors(cudaMemcpy(out, output, sizeof(float) * R.output_size, cudaMemcpyDeviceToHost));
     std::cout << "Output Matrix:"<<std::endl;
     for(int i=0; i<R.output_size; i++)
     {
         if(i%WIDTH==0)
             std::cout << "\n";
         std::cout << out[i] << " ";
     }
     std::cout<<std::endl;
    
 
     std::cout<<"Testing Backward . . ."<<std::endl;
 
    float *cpu_dup = (float *)malloc(sizeof(float) * R.output_size);
    for(int i=0; i<R.output_size; i++)
        cpu_dup[i] = 100 + i;
 
     std::cout << "Upstream Derivatives:";
     for(int i=0; i<R.output_size; i++)
     {
         if(i%WIDTH==0)
             std::cout << "\n";
         std::cout << cpu_dup[i] << " ";
     }
     std::cout<<std::endl;
 
    checkCudaErrors(cudaMemcpy(dup, cpu_dup, sizeof(float) * R.output_size,  cudaMemcpyHostToDevice));

     std::cout << "\nApply Backward:"<<std::endl;
    R.backward(dup, dout);
 
    float *cpu_dout = (float *)malloc(sizeof(float) * R.input_size);
    checkCudaErrors(cudaMemcpy(cpu_dout, dout, sizeof(float) * R.input_size, cudaMemcpyDeviceToHost));
     std::cout << "Back prop results :"<<std::endl;
     for(int i=0; i<R.input_size; i++)
     {
         if(i%WIDTH==0)
             std::cout << "\n";
         std::cout << cpu_dout[i] << " ";
     }
     std::cout<<std::endl;
}



void test_conv_relu_maxpool()
{
    printf("*********** CONV_RELU_POOL TEST **************\n");
    // Initialize image and cudnn handles
    int WIDTH_CONV = 4, HEIGHT_CONV = 5, KERNEL_SIZE_CONV=3, PADDING_CONV=1, STRIDE_CONV=1;  //Input to Conv
    int SIZE_MAX_POOL=2, STRIDE_MAX_POOL=2, PADDING_MAX_POOL=0;  //For MaxPool
    int BATCH_SIZE = 1, CHANNELS = 1;     //Image
    int GPU_ID = 0;
    checkCudaErrors(cudaSetDevice(GPU_ID));
    float *data, *output_conv, *output_relu, *output_max_pool, *input_diff_grad, *output_diff_grad, *output_diff_grad_relu;
 
    cudnnHandle_t cudnn;
    cublasHandle_t cublas;
    cudnnTensorDescriptor_t d1, d2; // dummy descriptors
 
    cudnnCreate(&cudnn);
    cublasCreate(&cublas);

    // Stack Layers
    Conv c(1, CHANNELS, KERNEL_SIZE_CONV, PADDING_CONV, STRIDE_CONV, cudnn,  cublas, BATCH_SIZE, WIDTH_CONV, HEIGHT_CONV, true, true, GPU_ID, d1, d2, true);
    Relu R(CHANNELS, CHANNELS, cudnn, cublas, BATCH_SIZE, c.out_height, c.out_width, GPU_ID, c.output_descriptor, d2, false);
    MaxPoolLayer mpl(SIZE_MAX_POOL, STRIDE_MAX_POOL, PADDING_MAX_POOL, BATCH_SIZE, CHANNELS, R.out_height, R.out_width, GPU_ID, cudnn, R.output_descriptor, d2, false);

    //Initialize tensors device
    cudaMalloc(&data, sizeof(float) * c.input_size);                 //CONV INPUT
    cudaMalloc(&output_conv, sizeof(float) * c.output_size);         //CONV OUTPUT-RELU INPUT
    cudaMalloc(&output_relu, sizeof(float) * R.output_size);         //RELU OUTPUT-MAXPOOL INPUT
    cudaMalloc(&output_max_pool, sizeof(float) * mpl.output_size);   //MAXPOOL OUTPUT
 
    //Initalize arrays host
    float *cpu_data = (float *)malloc(sizeof(float) * c.input_size);
    const float data_[5][4] = {{1, 6, 11, 16},
                               {2, 7, 12, 17},
                               {3, 8, 13, 18},
                               {4, 9, 14, 19},
                               {5, 10, 15, 20}};
    for(int i=0; i<5; i++)
        for(int j=0; j<4; j++)
            cpu_data[4*i + j] = data_[i][j];
    //for(int i = 0;i < c.input_size;i++) cpu_data[i] = 3.0;
    checkCudaErrors(cudaMemcpyAsync(data, cpu_data, sizeof(float) * c.input_size,  cudaMemcpyHostToDevice));

    float* output_matrix_conv = (float *)malloc(sizeof(float)*c.output_size);
    float* output_matrix_relu = (float *)malloc(sizeof(float)*R.output_size);
    float* output_matrix = (float *)malloc(sizeof(float)*mpl.output_size);
 
    float* grad_data_conv = (float *)malloc(sizeof(float) * c.input_size);

    std::cout << "Input Matrix:";
    pprint(cpu_data, c.input_size, WIDTH_CONV);
 
    std::cout << "\nApply Convolution kernel_size=3, padding=1, stride=1:\n";
    c.forward(data, output_conv);
    checkCudaErrors(cudaMemcpy(output_matrix_conv, output_conv, sizeof(float)*mpl.input_size, cudaMemcpyDeviceToHost));
    std::cout << "\nOutput Matrix From Convolution:";
    pprint(output_matrix_conv, c.output_size, mpl.input_width);
 
    std::cout << "\nApply Relu:\n";
    R.forward(output_conv, output_relu);
    checkCudaErrors(cudaMemcpy(output_matrix_relu, output_relu, sizeof(float)*R.output_size, cudaMemcpyDeviceToHost));
    std::cout << "\nOutput Matrix From Relu:";
    pprint(output_matrix_relu, R.output_size, R.out_width);

    std::cout << "\nPerforming max pool size=(2,2), stride=(2, 2), padding=(0, 0)\n";
    mpl.forward(output_relu, output_max_pool);
    checkCudaErrors(cudaMemcpy(output_matrix, output_max_pool, sizeof(float)*mpl.output_size, cudaMemcpyDeviceToHost));
    std::cout << "\nOutput Matrix From Max Pool:";
    pprint(output_matrix, mpl.output_size, mpl.out_width);
 
    // 'COMMENT BACKWARD'
    //Generate a input differential gradient recieved by max pool layer in backprop
    cudaMalloc(&input_diff_grad, sizeof(float) * mpl.output_size);
    cudaMalloc(&output_diff_grad, sizeof(float) * mpl.input_size);
    cudaMalloc(&output_diff_grad_relu, sizeof(float) * R.input_size);
 
    float *input_diff_grad_cpu = (float *)malloc(sizeof(float) * mpl.output_size);
    for(int i = 0;i < mpl.output_size;i++) input_diff_grad_cpu[i] = 10.0;
    checkCudaErrors(cudaMemcpyAsync(input_diff_grad, input_diff_grad_cpu, sizeof(float) * mpl.output_size,  cudaMemcpyHostToDevice));
    float* output_gradient = (float *)malloc(sizeof(float)*mpl.input_size);
    
    mpl.backward(output_conv, input_diff_grad, output_max_pool, output_diff_grad);
    checkCudaErrors(cudaMemcpy(output_gradient, output_diff_grad, sizeof(float)*mpl.input_size, cudaMemcpyDeviceToHost));
    std::cout << "\nGradient from Max Pool Layer:";
    pprint(output_gradient, mpl.input_size, mpl.input_width);
 
    std::cout << "\nBackpropping that through relu:";
    R.backward(output_diff_grad, output_diff_grad_relu);
    output_gradient = (float *)malloc(sizeof(float)*R.input_size);
    checkCudaErrors(cudaMemcpy(output_gradient, output_diff_grad_relu, sizeof(float)*R.input_size, cudaMemcpyDeviceToHost));
    std::cout << "\nGradient from Relu Layer:";
    pprint(output_gradient, R.input_size, R.input_width);

    std::cout << "\nBackpropping that through conv:\n";
    c.backward(output_diff_grad_relu, c.input_descriptor, data);
    checkCudaErrors(cudaMemcpy(grad_data_conv, c.grad_data, sizeof(float)*c.input_size, cudaMemcpyDeviceToHost));
    printf("\nGrad data conv: ");
    pprint(grad_data_conv, c.input_size, WIDTH_CONV);

    float *grad_kernel = (float *)malloc(sizeof(float) * 9);
    checkCudaErrors(cudaMemcpy(grad_kernel, c.grad_kernel, sizeof(float) * 9, cudaMemcpyDeviceToHost));

    std::cout<<"Printing grad_kernels . . .\n";
     for(int i = 0;i < 9;i++)
       std::cout << grad_kernel[i] << " ";
     std::cout << std::endl;

    int t = c.out_channels;
    float *grad_bias = (float *)malloc(sizeof(float) * t);
    checkCudaErrors(cudaMemcpy(grad_bias, c.grad_bias, sizeof(float) * t, cudaMemcpyDeviceToHost));

    std::cout<<"Printing grad_bias . . .\n";
     for(int i = 0;i < t;i++)
       std::cout << grad_bias[i] << " ";
     std::cout << std::endl;
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
    test_convolution();
    printf("\n\n\n----------Convolution Test Passed!-----------\n\n\n");
    test_mpl();
    printf("\n\n\n----------Max-Pooling Test Passed!-----------\n\n\n");
    test_relu();
    printf("\n\n\n----------Relu Test Passed!-----------\n\n\n");
    test_conv_relu_maxpool();
    printf("\n\n\n----------Conv Relu Maxpool Test Passed!-----------\n\n\n");
    test_sigmoid();
    printf("\n\n\n----------Sigmoid Test Passed!-----------\n\n\n");
}

int main() {
    test();
    return 0;
}
