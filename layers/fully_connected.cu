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


/*
Error Descriptions : CUDNN_STATUS_BAD_PARAM : Some data is not given right
                     CUDNN_STATUS_NOT_SUPPORTED : Given combination of inputs (descriptors) does not work
*/

///

class FullyConnectedLayer {
public:
	// alpha and beta are scaling constants for the operations, use these default values
    const float alpha = 1.0f;
    const float beta = 0.0f;

    /* Tensor Descriptors for our operation */
    cudnnTensorDescriptor_t input_descriptor;
    cudnnTensorDescriptor_t output_descriptor;
    cudnnTensorDescriptor_t bias_descriptor;
    cudnnFilterDescriptor_t kernel_descriptor; // descriptor for the weight parameter
    cudnnConvolutionDescriptor_t convolution_descriptor; // descriptor for the operation
    cudnnConvolutionFwdAlgo_t convolution_algorithm; // descriptor for the algorithm to use
    cudnnHandle_t cudnn;
    cublasHandle_t cublas;

    size_t workspace = 0, tmpsize = 0;
    void* d_workspace{nullptr};
    size_t m_workspaceSize;

    cudnnConvolutionBwdFilterAlgo_t convbwfalgo; // used for computing gradient with respect to weight
    cudnnConvolutionBwdDataAlgo_t convbwdalgo; // used for computing gradient with respect to input
    bool falgo, dalgo; // if falgo, we compute gradient with respect to filter weight parameter, if dalgo, we compute gradient with respect to input

    /*** These variables are on GPU ***/
    // weights of the kernel and bias
    float *param_kernel;
    float *param_bias;

    // placeholders for gradients of parameters
    float *grad_kernel;
    float *grad_bias;
    float *grad_data; // gradient with respect input of convolution, Note : INPUT

    /*** These variables are on CPU ***/
    std::vector<float> cpu_param_kernel;
    std::vector<float> cpu_param_bias;

    /*** Definition variables we would be using ***/
    int input_size;
    int output_size;
    int out_height;
    int out_width;
    int gpu_id;
    int in_channels, kernel_size, out_channels;

    FullyConnectedLayer(int _in_channels, cudnnHandle_t _cudnn, cublasHandle_t _cublas,
         int batch_size, int width, int height, bool use_backward_filter, bool use_backward_data, int gpu_id,
         cudnnTensorDescriptor_t& _input_descriptor, cudnnTensorDescriptor_t& _output_descriptor, bool init_io_desc) {
        
    }

    void init_test_weights() {
    }

    void init_weights() {
    }

    void forward(float *d_input, float *d_output) {
    }

    void backward(float *data_grad_above, cudnnTensorDescriptor_t tensor_below, float *data_below) {
    }

    void updateWeights(float learning_rate) {
    }

};

void test_forward() {
}

void test() {
    test_forward();
}

int main() {
    test();
    return 0;
}
