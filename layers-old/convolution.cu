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

class Conv {
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

    Conv(int _in_channels, int _out_channels, int _kernel_size, int padding, int stride, cudnnHandle_t _cudnn, cublasHandle_t _cublas,
         int batch_size, int width, int height, bool use_backward_filter, bool use_backward_data, int gpu_id,
         cudnnTensorDescriptor_t& _input_descriptor, cudnnTensorDescriptor_t& _output_descriptor, bool init_io_desc) {
    	/*
		use_backward_filter : Whether to compute gradient with respect to filter weights
		use_backward_data : Whether to compute gradient with respect to input
		init_io_desc : If true, the input and output descriptors are initialized from scratch else they are used as `_input_descriptor` and `_output_descriptor` as passed to the function
    	*/
        // Assign Handles
        cudnn = _cudnn;
        cublas = _cublas;
        // Assign the GPU id to run on
        gpu_id = gpu_id;
        checkCudaErrors(cudaSetDevice(gpu_id));

        // Assign dimension values
        in_channels = _in_channels;
        out_channels = _out_channels;
        kernel_size = _kernel_size;
        out_width = ((width - kernel_size + 2*padding)/stride) + 1;
        out_height = ((height - kernel_size + 2*padding)/stride) + 1;

        /*** Forward Propagation Descriptors ***/
        // Input Tensor
        checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
        checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor, // Uses Tensor Descriptor
                                              /*format=*/CUDNN_TENSOR_NHWC,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/batch_size,
                                              /*channels=*/in_channels,
                                              /*image_height=*/height,
                                              /*image_width=*/width));

        // Output Tensor
        checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
        checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor, // Uses Tensor Descriptor
                                              /*format=*/CUDNN_TENSOR_NHWC,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*batch_size=*/batch_size,
                                              /*channels=*/out_channels,
                                              /*image_height=*/out_height,
                                              /*image_width=*/out_width));

        // Bias Tensor
        checkCUDNN(cudnnCreateTensorDescriptor(&bias_descriptor));
        checkCUDNN(cudnnSetTensor4dDescriptor(bias_descriptor,
                                              CUDNN_TENSOR_NHWC,
                                              CUDNN_DATA_FLOAT,
                                              1, out_channels,
                                              1, 1));


        // Kernel Tensor
        
        checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
        checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor, // Uses Kernel Descriptor
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                              /*format=*/CUDNN_TENSOR_NHWC,
                                              /*out_channels=*/out_channels,
                                              /*in_channels=*/in_channels,
                                              /*kernel_height=*/kernel_size,
                                              /*kernel_width=*/kernel_size));

        /*** Create Convolution Descriptors ***/
        
        checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
        checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                                   /*pad_height=*/padding,
                                                   /*pad_width=*/padding,
                                                   /*vertical_stride=*/stride,
                                                   /*horizontal_stride=*/stride,
                                                   /*dilation_height=*/1,
                                                   /*dilation_width=*/1,
                                                   /*mode=*/CUDNN_CONVOLUTION, // CUDNN_CROSS_CORRELATION,
                                                   /*computeType=*/CUDNN_DATA_FLOAT));

        /*** Create Convolution Algorithm Descriptors ***/
        if(init_io_desc)
            checkCUDNN(
                cudnnGetConvolutionForwardAlgorithm(cudnn,
                                                    input_descriptor,
                                                    kernel_descriptor,
                                                    convolution_descriptor,
                                                    output_descriptor,
                                                    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                    /*memoryLimitInBytes=*/0,
                                                    &convolution_algorithm));
        else
            checkCUDNN(
                cudnnGetConvolutionForwardAlgorithm(cudnn,
                                                    _input_descriptor, // init with what was passed to the function
                                                    kernel_descriptor,
                                                    convolution_descriptor,
                                                    _output_descriptor,
                                                    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                    /*memoryLimitInBytes=*/0,
                                                    &convolution_algorithm));


        // /*** Allocating Memory To Workspace for the operations ***/
        if(init_io_desc)
            checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                               input_descriptor,
                                                               kernel_descriptor,
                                                               convolution_descriptor,
                                                               output_descriptor,
                                                               convolution_algorithm,
                                                               &workspace));
        else
            checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                                   _input_descriptor, // init with what was passed to the function
                                                                   kernel_descriptor,
                                                                   convolution_descriptor,
                                                                   _output_descriptor,
                                                                   convolution_algorithm,
                                                                   &workspace));

        /*** Backward Propagation Descriptors ***/
        // set falgo and dalgo
        falgo = use_backward_filter;
        dalgo = use_backward_data;
        //

        // If backprop filter algorithm was requested
        if (falgo)
        {   
            if(init_io_desc) {
                checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(
                    cudnn, input_descriptor, output_descriptor, convolution_descriptor, kernel_descriptor,
                    CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &convbwfalgo));

                checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                    cudnn, input_descriptor, output_descriptor, convolution_descriptor, kernel_descriptor, 
                    convbwfalgo, &tmpsize));
            }
            else {
                checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(
                cudnn, _input_descriptor, _output_descriptor, convolution_descriptor, kernel_descriptor,
                CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &convbwfalgo));

                checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                    cudnn, _input_descriptor, _output_descriptor, convolution_descriptor, kernel_descriptor, 
                    convbwfalgo, &tmpsize));
            }
        }

        workspace = std::max(workspace, tmpsize);

        // // If backprop data algorithm was requested
        if (dalgo)
        {
            if(init_io_desc) {
                checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
                    cudnn, kernel_descriptor, output_descriptor, convolution_descriptor, input_descriptor,
                    CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &convbwdalgo));

                checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
                    cudnn, kernel_descriptor, output_descriptor, convolution_descriptor, input_descriptor, 
                    convbwdalgo, &tmpsize));
            }
            else {
                checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
                    cudnn, kernel_descriptor, _output_descriptor, convolution_descriptor, _input_descriptor,
                    CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &convbwdalgo));

                checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
                    cudnn, kernel_descriptor, _output_descriptor, convolution_descriptor, _input_descriptor, 
                    convbwdalgo, &tmpsize));
            }
        }

        workspace = std::max(workspace, tmpsize);

        cudaMalloc(&d_workspace, workspace);
        m_workspaceSize = workspace;

        /*** Allocate memory to kernel and bias ***/
        checkCudaErrors(cudaMalloc(&param_kernel, sizeof(float) * in_channels * kernel_size * kernel_size * out_channels));
        checkCudaErrors(cudaMalloc(&param_bias, sizeof(float) * out_channels));
        checkCudaErrors(cudaMalloc(&grad_kernel, sizeof(float) * in_channels * kernel_size * kernel_size * out_channels));
        checkCudaErrors(cudaMalloc(&grad_bias, sizeof(float) * out_channels));
        // Gradient with respect to output has same shape as output
        checkCudaErrors(cudaMalloc(&grad_data,   sizeof(float) * batch_size * out_height * out_width * out_channels));

        input_size = batch_size * height * width * in_channels;
        output_size = batch_size * out_height * out_width * out_channels;

        // Initialie CPU-parameter memory
        cpu_param_kernel = std::vector<float>(in_channels * kernel_size * kernel_size * out_channels, 0);
        cpu_param_bias = std::vector<float>(out_channels, 0);

        // Initialize Parameters on GPU
        // init_weights();
        init_test_weights();

        // Move Initialized Weights to GPU
        // checkCudaErrors(cudaMemcpyAsync(param_kernel, &cpu_param_kernel[0],     sizeof(float) * cpu_param_kernel.size(),  cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpyAsync(param_bias, &cpu_param_bias[0], sizeof(float) * cpu_param_bias.size(),  cudaMemcpyHostToDevice));
    }

    void init_test_weights() {
    	// Allocate Kernel
		const float kernel_template[2][2] = {
		  {1, 3},
		  {2, 4}
		};

		float h_kernel[1][2][2][1];
	    for (int row = 0; row < 2; ++row)
	      for (int column = 0; column < 2; ++column)
	        h_kernel[0][row][column][0] = kernel_template[row][column];

		cudaMemcpy(param_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);
    }

    void init_weights() {
    	// Initialize Weights
        std::random_device rd;
        std::mt19937 gen(RANDOM_SEED < 0 ? rd() : static_cast<unsigned int>(RANDOM_SEED));

        // Xavier Initialization
        float wconv = sqrt(3.0f / (kernel_size * kernel_size * in_channels));
        std::uniform_real_distribution<> dconv(-wconv, wconv);
        for (auto&& iter : cpu_param_kernel)
            iter = static_cast<float>(dconv(gen));
    }

    void forward(float *d_input, float *d_output) {
        checkCudaErrors(cudaSetDevice(gpu_id));
        checkCUDNN(cudnnConvolutionForward(cudnn,
                                       &alpha,
                                       input_descriptor,
                                       d_input,
                                       kernel_descriptor,
                                       param_kernel,
                                       convolution_descriptor,
                                       convolution_algorithm,
                                       d_workspace,
                                       m_workspaceSize,
                                       &beta,
                                       output_descriptor,
                                       d_output));

        checkCUDNN(cudnnAddTensor(cudnn, &alpha, bias_descriptor,
                                  param_bias, &alpha, output_descriptor, d_output));
    }

    void backward(float *data_grad_above, cudnnTensorDescriptor_t tensor_below, float *data_below) {
    	/*
		X : Input
		Y : Output
		W,b : Convolution Parameters

		Y = WX + b
		Y : Output of given convolution
		
		This calculates dW, db, dX

		data_grad_above : dY
		tensor_below : Descriptor of X
		data_below : X
    	*/
        checkCudaErrors(cudaSetDevice(gpu_id));
        checkCUDNN(cudnnConvolutionBackwardBias(cudnn, &alpha, output_descriptor,
                                                data_grad_above, &beta, bias_descriptor, grad_bias)); // correct!

        if(falgo)
            checkCUDNN(cudnnConvolutionBackwardFilter(cudnn, &alpha, tensor_below,
                                                      data_below, output_descriptor, data_grad_above, convolution_descriptor,
                                                      convbwfalgo, d_workspace, m_workspaceSize,
                                                      &beta, kernel_descriptor, grad_kernel)); // workspace ka dekhna, baaki correct hai!
        
        if(dalgo)
            checkCUDNN(cudnnConvolutionBackwardData(cudnn, &alpha, kernel_descriptor,
                                                    param_kernel, output_descriptor, data_grad_above, convolution_descriptor,
                                                    convbwdalgo, d_workspace, m_workspaceSize,
                                                    &beta, tensor_below, grad_data));
    }

    void updateWeights(float learning_rate) {
        int ks = in_channels * kernel_size * kernel_size * out_channels;
        int bs = out_channels;
        checkCudaErrors(cublasSaxpy(cublas, static_cast<int>(ks),
                                    &alpha, grad_kernel, 1, param_kernel, 1));
        checkCudaErrors(cublasSaxpy(cublas, static_cast<int>(bs),
                                    &alpha, grad_bias, 1, param_bias, 1));
    }

    void save_params(const char* fileprefix) {

        // get full filenames from the file prefix provided
        std::string param_kernel_file = std::string(fileprefix) + ".bin";
        std::string param_bias_file = std::string(fileprefix) + ".bias.bin";
        
        // Writing the weights to the file
        FILE *fp = fopen(param_kernel_file.c_str(), "wb");
        if (!fp) {
            printf("FILE ERROR: Cannot open file %s\n", param_kernel_file.c_str());
            exit(2);
        }
        fwrite(&cpu_param_kernel[0], sizeof(float), in_channels * kernel_size * kernel_size * out_channels, fp);
        fclose(fp);
    
        // Write the bias to the file
        fp = fopen(param_bias_file.c_str(), "wb");
        if (!fp) {
            printf("FILE ERROR: Cannot open file %s\n", param_bias_file.c_str());
            exit(2);
        }
        fwrite(&cpu_param_bias[0], sizeof(float), out_channels, fp);
        fclose(fp);
    
    }

    bool load_params(const char* fileprefix) {
        
        // get full filenames from the file prefix provided
        std::string param_kernel_file = std::string(fileprefix) + ".bin";
        std::string param_bias_file = std::string(fileprefix) + ".bias.bin";
        
        // reading the weights from the file
        FILE *fp = fopen(param_kernel_file.c_str(), "rb");
        if (!fp) {
            printf("FILE ERROR: Cannot open file %s\n", param_kernel_file.c_str());
            return false;
        }
        fread(&cpu_param_kernel[0], sizeof(float), in_channels * kernel_size * kernel_size * out_channels, fp);
        fclose(fp);
    
        // reading the bias from the file
        fp = fopen(param_bias_file.c_str(), "rb");
        if (!fp) {
            printf("FILE ERROR: Cannot open file %s\n", param_bias_file.c_str());
            return false;
        }
        fread(&cpu_param_bias[0], sizeof(float), out_channels, fp);
        fclose(fp);
    
        return true;
    
    }

};

void test_forward() {
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
	
    c.forward(data, output);

    // Move from device to host
    float *out = (float *)malloc(sizeof(float) * c.output_size);
    // float out[BATCH_SIZE][c.out_height][c.out_width][c.out_channels];
    checkCudaErrors(cudaMemcpy(out, output, sizeof(float) * c.output_size, cudaMemcpyDeviceToHost));

    // for(int i = 0;i < c.out_height;i++) {
    // 	for(int j = 0;j < c.out_width;j++)
    // 		std::cout << out[0][i][j][0] << " ";
    // 	std::cout << std::endl;
    // }
    // std::cout << std::endl;
    for(int i = 0;i < c.output_size;i++) {
    	std::cout << out[i] << " ";
    }
    std::cout << std::endl;
    std::cout << c.output_size << std::endl;
}

void test() {
    test_forward();
}

int main() {
    test();
    return 0;
}
