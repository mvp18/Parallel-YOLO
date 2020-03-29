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
    cudnnTensorDescriptor_t input_tensor;
    cudnnTensorDescriptor_t output_tensor;

    cublasHandle_t cublas;

    /*** These variables are on GPU ***/
    // weights and bias
    float *weights;
    float *bias;

    /*** These variables are on CPU ***/
    std::vector<float> cpu_weights;
    std::vector<float> cpu_bias;

    int gpu_id;

    FullyConnectedLayer(int input_size, int output_size, int batch_size, cublasHandle_t _cublas, int gpu_id,
         cudnnTensorDescriptor_t& _input_descriptor, cudnnTensorDescriptor_t& _output_descriptor) {
        
        // Create tensor for input (output from the pooling layer)
        checkCUDNN(cudnnCreateTensorDescriptor(&input_tensor));
        
        // Create tensor for output
        checkCUDNN(cudnnCreateTensorDescriptor(&output_tensor));

        // Set tensor description
        checkCUDNN(cudnnSetTensor4dDescriptor(input_tensor,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            batch_size, input_size, 1, 1));

        checkCUDNN(cudnnSetTensor4dDescriptor(output_tensor,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            batch_size, output_size, 1, 1));

    }

    void init_test_weights() {
    }

    void init_weights() {
    }

    void forward(float *input_data, float *output_data, float, float *onevec) {

        // Forward propagation using weights
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
            output_size, batch_size, input_size,
            &alpha,
            weights, input_size,
            input_data, input_size,
            &beta,
            output_data, output_size));

        // Adding bias to output_data
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
            output_size, batch_size, 1,
            &alpha,
            bias, output_size,
            onevec, 1,
            &alpha,
            output_data, output_size));

    }

    void backward(float *data_grad_above, float *data_below, float* onevec) {
    }

    void updateWeights(float learning_rate) {
`       
        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc1.pneurons.size()),
        &alpha, gfc1, 1, pfc1, 1));

        checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc1.pbias.size()),
        &alpha, gfc1bias, 1, pfc1bias, 1));
    
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
