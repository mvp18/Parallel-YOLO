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

class MaxPoolLayer{
public:
    float alpha = 1.0f, beta = 0.0f;
    int gpu_id, input_height, input_width,  input_size, output_height, output_width, output_size;

    cudnnPoolingDescriptor_t poolDesc;
    cudnnTensorDescriptor_t input_descriptor, poolTensor;

    cudnnHandle_t cudnnHandle;

    MaxPoolLayer(int size, int stride, int padding, int batch_size, int conv_out_channel, int conv_out_height, int conv_out_width, int _gpu_id, cudnnHandle_t _cudnnHandle){

        // Assign Handles
        cudnnHandle=_cudnnHandle;
    
        // Assign the GPU id to run on
        gpu_id = _gpu_id;
        checkCudaErrors(cudaSetDevice(gpu_id));

        /*** Forward Propagation Descriptors ***/
        input_width = conv_out_width;
        input_height = conv_out_height;
        input_size = input_height*input_width;
        output_height = (conv_out_height-size+2*padding)/stride + 1;
        output_width = (conv_out_width-size+2*padding)/stride + 1;
        output_size = output_height*output_width;
        // Input Tensor (it is the output tensor from the convolution layer)
        checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
        checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,                 // Uses Tensor Descriptor
                                              CUDNN_TENSOR_NHWC,                //format
                                              CUDNN_DATA_FLOAT,                 //dataType
                                              batch_size,                       //batch_size
                                              conv_out_channel,                 //channels
                                              conv_out_height,                  //image_height
                                              conv_out_width));                 //image_width

        // Pooling Descriptor
        checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));            
        checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc,
                                               CUDNN_POOLING_MAX,
                                               CUDNN_PROPAGATE_NAN,
                                               size, size,
                                               padding, padding,
                                               stride, stride));
        // Output Tensor
        checkCUDNN(cudnnCreateTensorDescriptor(&poolTensor));
        checkCUDNN(cudnnSetTensor4dDescriptor(poolTensor,
                                            CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT,
                                            batch_size, 
                                            conv_out_channel,
                                            output_height,
                                            output_width));

        
    }

    void forward(float* input_tensor, float* output_tensor){
        checkCudaErrors(cudaSetDevice(gpu_id));
        checkCUDNN(cudnnPoolingForward(cudnnHandle,         //handle
                                       poolDesc,            //poolingDesc
                                       &alpha,              //alpha
                                       input_descriptor,    //xDesc
                                       input_tensor,        //x
                                       &beta,               //beta
                                       poolTensor,          //yDesc
                                       output_tensor));     //y    
    }

    void backward(float *data_grad_from_above_conv, float *data_grad_below, cudnnTensorDescriptor_t output_tensor_of_below_conv, float *data_grad_above_from_below_conv, float* data_grad_below_from_below_conv, float* data_below, float* input_tensor, float* output_tensor){
        checkCUDNN(cudnnPoolingBackward(cudnnHandle,        //handle
                                        poolDesc,           //poolingDesc
                                        &alpha,             //alpha
                                        poolTensor,         //yDesc
                                        output_tensor,               //y
                                        poolTensor,         //dyDesc
                                        data_grad_from_above_conv,    //dy 
                                        output_tensor_of_below_conv,       //xDesc
                                        input_tensor,         //x     (NOT SURE ABOUT THIS)
                                        &beta,                             //beta
                                        output_tensor_of_below_conv,         //dxDesc
                                        data_grad_below_from_below_conv));                              //dx
    }

    void update_weights(){
        // No weights
        return;
    }
};

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
    
    std::cout << "Input Matrix:\n";
    for(int i=0; i<mpl.input_size; i++){
    if(i%WIDTH==0) std::cout << "\n";
    std::cout << input_matrix[i] << " ";
  }
  
  std::cout << "\n\nPerforming max pool Size\n";
  mpl.forward(data, output);
  
  checkCudaErrors(cudaMemcpy(output_matrix, output, sizeof(float)*mpl.output_size, cudaMemcpyDeviceToHost));
  std::cout << "\nOutput Matrix:\n";
  for(int i=0; i<mpl.output_size; i++){
    if(i%mpl.output_width==0) std::cout << "\n";
    std::cout << output_matrix[i] << " ";
  }
  std::cout << "\n";
}


int main() {
	test_forward_mpl();
	return 0;
}