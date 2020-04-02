%%cuda --name cudnn.cu
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
//#include <cublas_v2.h>
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

class Relu
{
    public:
 
        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        cudnnTensorDescriptor_t input_descriptor;
        cudnnTensorDescriptor_t output_descriptor;

        cudnnActivationDescriptor_t activation_descriptor;
        
        cudnnHandle_t cudnn;
        //cublasHandle_t cublas;
 
        /*** These variables will be on GPU as cache for backward pass ***/
        float *din; //Input to ReLU layer
        float *dot;  //Output of ReLU layer
 
        /*** These variables will be on CPU ***/
        int input_size, output_size;
        int out_height, out_width;
        int in_channels, out_channels;
        int gpu_id;
        float *din_cpu; //Cache for backprop
        float *dot_cpu; //Cache for backprop
 
        Relu(int _in_channels, int _out_channels, cudnnHandle_t _cudnn, /*cublasHandle_t _cublas,*/
             int batch_size, int height, int width, int _gpu_id)
        {
            cudnn = _cudnn;
            //cublas = _cublas;
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
            
        
            checkCUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
            checkCUDNN(cudnnSetActivationDescriptor(activation_descriptor,
                                         CUDNN_ACTIVATION_RELU,
                                         CUDNN_NOT_PROPAGATE_NAN,
                                         0.0)); //Clip value for Clipped Relu, Not of any use here
         
            /*** Allocate memory to GPU and CPU ***/
            input_size = batch_size * in_channels * height * width;
            output_size = input_size;
         
            din_cpu = (float *)malloc(sizeof(float) * input_size);
            dot_cpu = (float *)malloc(sizeof(float) * output_size);
         
            checkCudaErrors(cudaMalloc(&din, sizeof(float)*input_size));
            checkCudaErrors(cudaMalloc(&dot, sizeof(float)*output_size));
        }
 
        void forward(float *d_input, float *d_output)
        {
            checkCUDNN(cudnnActivationForward(
                cudnn,
                activation_descriptor,
                &alpha,
                input_descriptor,
                d_input,
                &beta,
                output_descriptor,
                d_output
            ));

            ///Store Input Output in Cache for backprop
            checkCudaErrors(cudaMemcpy(dot_cpu, d_output, sizeof(float) * output_size, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(dot, dot_cpu, sizeof(float) * output_size,  cudaMemcpyHostToDevice));
         
            checkCudaErrors(cudaMemcpy(din_cpu, d_input, sizeof(float) * output_size, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(din, din_cpu, sizeof(float) * output_size,  cudaMemcpyHostToDevice));

        }
 
        void backward(float *grad_above, float *grad_out/*, float *d_input, float *d_output*/) //
        {
            checkCUDNN(cudnnActivationBackward(
                cudnn,
                activation_descriptor,
                &alpha,
                output_descriptor,
                dot,
                output_descriptor,
                grad_above,
                input_descriptor,
                din, //Not sure why this parameter is required!
                &beta,
                input_descriptor,
                grad_out
            ));
        }
};

void test_relu() 
{
    int WIDTH = 5, HEIGHT = 5, BATCH_SIZE = 1, CHANNELS = 1;
    int GPU_ID = 0;
    checkCudaErrors(cudaSetDevice(GPU_ID));
 
    float *data, *output, *dup, *dout;
    cudnnHandle_t cudnn;
    //cublasHandle_t cublas;

    //cudnnTensorDescriptor_t d1, d2; // dummy descriptors
    cudnnCreate(&cudnn);
    //cublasCreate(&cublas);
 
    Relu R(CHANNELS, CHANNELS, cudnn, /*cublas,*/ BATCH_SIZE, HEIGHT, WIDTH, GPU_ID);
 
    cudaMalloc((void **)&data, sizeof(float) * R.input_size);
    cudaMalloc((void **)&output, sizeof(float) * R.output_size);
    cudaMalloc((void **)&dout, sizeof(float) * R.output_size);
    cudaMalloc((void **)&dup, sizeof(float) * R.output_size);

    float *cpu_data = (float *)malloc(sizeof(float) * R.input_size);
    for(int i = 0; i < R.input_size; i++)
        cpu_data[i] = -12.0 + i;
    cpu_data[1] = 3234.0; //to check clipping
    cpu_data[20] = 3566.0;
 
    cout<<"Testing Forward . . ."<<endl;
 
    cout << "Input Matrix:"<<endl;
    for(int i=0; i<R.input_size; i++)
    {
        if(i%WIDTH==0)
            cout << "\n";
        cout << cpu_data[i] << " ";
    }
    cout << "\nApply ReLU:"<<endl;
 
    checkCudaErrors(cudaMemcpy(data, cpu_data, sizeof(float) * R.input_size,  cudaMemcpyHostToDevice));
    cout << "\nApply ReLU 2:"<<endl;
    R.forward(data, output);
 
    float *out = (float *)malloc(sizeof(float) * R.output_size);
    checkCudaErrors(cudaMemcpy(out, output, sizeof(float) * R.output_size, cudaMemcpyDeviceToHost));
    cout << "Output Matrix:"<<endl;
    for(int i=0; i<R.output_size; i++)
    {
        if(i%WIDTH==0)
            cout << "\n";
        cout << out[i] << " ";
    }
    cout<<endl;
 
 
    cout<<"Testing Backward . . ."<<endl;
 
    float *cpu_dup = (float *)malloc(sizeof(float) * R.output_size);
    for(int i=0; i<R.output_size; i++)
        cpu_dup[i] = 100 + i;
 
    cout << "Upstream Derivatives:";
    for(int i=0; i<R.output_size; i++)
    {
        if(i%WIDTH==0)
            cout << "\n";
        cout << cpu_dup[i] << " ";
    }
    cout<<endl;
 
    checkCudaErrors(cudaMemcpy(dup, cpu_dup, sizeof(float) * R.output_size,  cudaMemcpyHostToDevice));

    cout << "\nApply Backward:"<<endl;
    R.backward(dup, dout);
 
    float *cpu_dout = (float *)malloc(sizeof(float) * R.input_size);
    checkCudaErrors(cudaMemcpy(cpu_dout, dout, sizeof(float) * R.input_size, cudaMemcpyDeviceToHost));
    cout << "Back prop results :"<<endl;
    for(int i=0; i<R.input_size; i++)
    {
        if(i%WIDTH==0)
            cout << "\n";
        cout << cpu_dout[i] << " ";
    }
    cout<<endl;
}

int main()
{
    cout<<"In main function . . ."<<endl;
    test_relu();
    return 0;
}