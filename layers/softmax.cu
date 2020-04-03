%%cuda --name softmax.cu
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

#include <cuda.h>
#include <cuda_runtime.h>

#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <cudnn.h>

using namespace std;

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


class Softmax 
/*Expected Input Tensor Shape [N,1,1,W] in NCHW format in constructor
N = BATCH_SIZE, W = flattened vector length
Output is of same shape out = softmax(inp) 
backprop expects dL/dout (grad_in) and returns dL/dinp
L = any loss computed from output of softmax eg cross entropy 
Note that we need to have another layer to compute loss if we like to calculate*/
{
    public:
 
        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        cudnnTensorDescriptor_t input_descriptor;
        cudnnTensorDescriptor_t output_descriptor;
        
        cudnnHandle_t cudnn;
        cublasHandle_t cublas;
 
        /*** These variables will be on GPU as cache for backward pass ***/
        float *dot;  //Output of softmax i.e., dot = softmax(d_input) in forward, necessary to cache for backward
 
        /*** These variables will be on CPU ***/
        int input_size, output_size;
        int out_height, out_width;
        int in_channels, out_channels;
        int gpu_id;
        float *dot_cpu; //Cache for backprop
 
        Softmax(int _in_channels, int _out_channels, cudnnHandle_t _cudnn, cublasHandle_t _cublas,
             int batch_size, int height, int width, int _gpu_id)
        {
            cudnn = _cudnn;
            cublas = _cublas;
            gpu_id = _gpu_id;

            checkCudaErrors(cudaSetDevice(gpu_id));
         
            in_channels = _in_channels;
            out_channels = _out_channels;
            out_width = width;
            out_height = height;
         
            checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
            checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor, 
                                                      CUDNN_TENSOR_NCHW,
                                                      CUDNN_DATA_FLOAT,
                                                      batch_size,
                                                      in_channels,
                                                      height,
                                                      width));
                
            checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
            checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                                      CUDNN_TENSOR_NCHW,
                                                      CUDNN_DATA_FLOAT,
                                                      batch_size,
                                                      out_channels,
                                                      out_height,
                                                      out_width));
            
            /*** Allocate memory to GPU placeholders ***/
            input_size = batch_size * in_channels * height * width;
            output_size = input_size; //output_size means output of softmax, not the scalar loss
         
            checkCudaErrors(cudaMalloc(&dot, sizeof(float) * output_size));
            dot_cpu = (float *)malloc(sizeof(float) * output_size);
        }
 
        void forward(float *d_input, float *d_output)
        {
            checkCUDNN(cudnnSoftmaxForward(
                cudnn,
                CUDNN_SOFTMAX_ACCURATE,
                CUDNN_SOFTMAX_MODE_INSTANCE,
                &alpha,
                input_descriptor,
                d_input,
                &beta,
                output_descriptor,
                d_output
            ));
         
            //Store the output of softmax for backprop
            checkCudaErrors(cudaMemcpy(dot_cpu, d_output, sizeof(float) * output_size, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(dot, dot_cpu, sizeof(float) * output_size,  cudaMemcpyHostToDevice));
        }
 
        void backward(float *grad_above, float *grad_out)
        {
            checkCUDNN(cudnnSoftmaxBackward(
                cudnn,
                CUDNN_SOFTMAX_ACCURATE,
                CUDNN_SOFTMAX_MODE_INSTANCE,
                &alpha,
                output_descriptor,
                dot,
                output_descriptor,
                grad_above,
                &beta,
                input_descriptor,
                grad_out
            ));
        }
};

void pprint(float *a, int n, int WIDTH)
{
    for(int i=0; i<n; i++)
    {
        if(i % WIDTH==0)
            cout << "\n";
        cout << a[i] << " ";
    }
    cout<<endl;
}

void test_softmax() 
{
    int WIDTH = 5, HEIGHT = 1, BATCH_SIZE = 5, CHANNELS = 1; //Input to softmax is of shape (N,1,1,W)
    int GPU_ID = 0;
    checkCudaErrors(cudaSetDevice(GPU_ID));
 
    float *data, *dout;
    cudnnHandle_t cudnn;
    cublasHandle_t cublas;

    cudnnCreate(&cudnn);
    cublasCreate(&cublas);
 
    Softmax R(CHANNELS, CHANNELS, cudnn, cublas, BATCH_SIZE, HEIGHT, WIDTH, GPU_ID);
 
    cudaMalloc((void **)&data, sizeof(float) * R.input_size);
    cudaMalloc((void **)&dout, sizeof(float) * R.output_size);
 
    //cudaMalloc((void **)&dtarget, sizeof(float) * R.output_size);

    float *cpu_data = (float *)malloc(sizeof(float) * R.input_size);
    //float *cpu_target = (float *)malloc(sizeof(float) * R.input_size);
    //float *cpu_loss = (float *)malloc(sizeof(float) * 1);
    for(int i = 0; i < R.input_size; i++)
    {
        cpu_data[i] = i+1.0;
    }
    cpu_data[5] = 1;
    cpu_data[20] = -1;
 
    cout<<"Testing Softmax forward . . ."<<endl;
    cout << "Input Matrix:";
    pprint(cpu_data, R.input_size, WIDTH);
    //cout<<"Target :";
    //pprint(cpu_target, R.input_size, WIDTH);
 
    cout << "\nApply Softmax:"<<endl;
    checkCudaErrors(cudaMemcpy(data, cpu_data, sizeof(float) * R.input_size,  cudaMemcpyHostToDevice));
    //checkCudaErrors(cudaMemcpy(dtarget, cpu_target, sizeof(float) * R.input_size,  cudaMemcpyHostToDevice));
    
    R.forward(data, dout);
 
    float *out = (float *)malloc(sizeof(float) * R.output_size);
    checkCudaErrors(cudaMemcpy(out, dout, sizeof(float) * R.output_size, cudaMemcpyDeviceToHost));
    //checkCudaErrors(cudaMemcpy(cpu_loss, dloss, sizeof(float) * R.output_size, cudaMemcpyDeviceToHost));
    //cout<<"Loss = "<<cpu_loss[0]<<endl;
    cout << "Output Matrix:";
    pprint(out, R.output_size, WIDTH);
 
 
    cout<<"Testing Backward . . ."<<endl;
    float *cpu_dup = (float *)malloc(sizeof(float) * R.output_size);
    for(int i=0; i<R.output_size; i++)
        cpu_dup[i] = 0;
 
    //Remember dL/dy_hat = [0, 0, 0, 0 . . . , -1/y_hat[k], 0, 0, . . ., 0]
    cpu_dup[2] = -1 / out[2]; //It means 1st row in Batch had target label at index = 2
    cpu_dup[8] = -1 / out[8]; //It means 1st row in Batch had target label at index = 8 and so on
    cpu_dup[10] = -1/out[10];
    cpu_dup[16] = -1/out[16];
    cpu_dup[22] = -1/out[22];

 
    cout << "Upstream Derivatives:";
    pprint(cpu_dup, R.output_size, WIDTH);
 
    float *dup, *dgrad;
    cudaMalloc((void **)&dup, sizeof(float) * R.output_size);
    cudaMalloc((void **)&dgrad, sizeof(float) * R.input_size);
 
    checkCudaErrors(cudaMemcpy(dup, cpu_dup, sizeof(float) * R.output_size,  cudaMemcpyHostToDevice));
    cout << "\nApply Backward:"<<endl;
    R.backward(dup, dgrad);
 
    float *cpu_dout = (float *)malloc(sizeof(float) * R.input_size);
    checkCudaErrors(cudaMemcpy(cpu_dout, dgrad, sizeof(float) * R.input_size, cudaMemcpyDeviceToHost));
    cout << "Back prop results (Expected y_hat - y_target for each row):"<<endl;
    pprint(cpu_dout, R.input_size, WIDTH);
 
    cout<<endl;
}

int main()
{
    cout<<"In main function . . ."<<endl;
    test_softmax();
    return 0;
}