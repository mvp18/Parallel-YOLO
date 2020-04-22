#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cfloat>
#include<stdlib.h>
#include<stdio.h>

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

#include<png.h>

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

#include "../include/convolution.h"
#include "../include/max_pool.h"
#include "../include/relu.h"
#include "../include/sigmoid.h"
#include "../include/data_utils.h"

void pprint(float* matrix, int size, int width)
{
    for(int i=0; i<size; i++)
    {
        printf("%.2f ", matrix[i]);
        if(i%width == 0 && i != 0)
            printf("\n");
    }
    std::cout << std::endl;
}

int main()
{
    int input_height, input_width, in_channels;
    int num_classes, num_anchors;
    int batch_size;
    //float learning_rate;
    int epochs;
    int ITERS;
    int num_images;
    int GPU_ID = 0;
    
    /* Read from config file */
    input_height = input_width = 416;
    in_channels = 3;
    batch_size = 1;
    num_classes = 1;
    num_anchors = 5;
    num_images = 1;
    epochs = 1;
    ITERS = epochs * num_images;

    /* Initialise few variables */
    int input_size = batch_size * in_channels * input_height * input_width;
    int final_output_depth = num_anchors * (num_classes + 5);
 
    checkCudaErrors(cudaSetDevice(GPU_ID));
    cudnnHandle_t cudnn;
    cublasHandle_t cublas;
    cudnnCreate(&cudnn);
    cublasCreate(&cublas);
    cudnnTensorDescriptor_t d1, d2; // dummy descriptors
 
    /* Define the Model */
    Conv c1(in_channels, 192, 13, 1, 2, cudnn, cublas, batch_size, input_width, input_height, true, false, GPU_ID, d1, d2, true); //kernel = 13, padding = 1, stride = 2
    Relu r1(c1.out_channels, c1.out_channels, cudnn, cublas, batch_size, c1.out_height, c1.out_width, GPU_ID, c1.output_descriptor, d2, false);
    MaxPoolLayer m1(2, 2, 0, batch_size, r1.out_channels, r1.out_height, r1.out_width, GPU_ID, cudnn, r1.output_descriptor, d2, false); //size = 2, stride = 2, padding = 0
 
    Conv c2(m1.out_channels, 256, 3, 1, 1, cudnn, cublas, batch_size, m1.out_width, m1.out_height, true, true, GPU_ID, m1.poolTensor, d2, false);
    Relu r2(c2.out_channels, c2.out_channels, cudnn, cublas, batch_size, c2.out_height, c2.out_width, GPU_ID, c2.output_descriptor, d2, false);
    MaxPoolLayer m2(2, 2, 0, batch_size, r2.out_channels, r2.out_height, r2.out_width, GPU_ID, cudnn, r2.output_descriptor, d2, false);
 
    Conv c3(m2.out_channels, 512, 3, 1, 1, cudnn, cublas, batch_size, m2.out_width, m2.out_height, true, true, GPU_ID, m2.poolTensor, d2, false);
    Relu r3(c3.out_channels, c3.out_channels, cudnn, cublas, batch_size, c3.out_height, c3.out_width, GPU_ID, c3.output_descriptor, d2, false);
 
    Conv c4(r3.out_channels, 1024, 1, 0, 1, cudnn, cublas, batch_size, r3.out_width, r3.out_height, true, true, GPU_ID, r3.output_descriptor, d2, false);
    Relu r4(c4.out_channels, c4.out_channels, cudnn, cublas, batch_size, c4.out_height, c4.out_width, GPU_ID, c4.output_descriptor, d2, false);
 
    Conv c5(r4.out_channels, 512, 3, 1, 1, cudnn, cublas, batch_size, r4.out_width, r4.out_height, true, true, GPU_ID, r4.output_descriptor, d2, false);
    Relu r5(c5.out_channels, c5.out_channels, cudnn, cublas, batch_size, c5.out_height, c5.out_width, GPU_ID, c5.output_descriptor, d2, false);
    MaxPoolLayer m5(2, 2, 1, batch_size, r5.out_channels, r5.out_height, r5.out_width, GPU_ID, cudnn, r5.output_descriptor, d2, false);
 
    Conv c6(m5.out_channels, 256, 3, 1, 1, cudnn, cublas, batch_size, m5.out_width, m5.out_height, true, true, GPU_ID, m5.poolTensor, d2, false);
    Relu r6(c6.out_channels, c6.out_channels, cudnn, cublas, batch_size, c6.out_height, c6.out_width, GPU_ID, c6.output_descriptor, d2, false);
    MaxPoolLayer m6(2, 2, 0, batch_size, r6.out_channels, r6.out_height, r6.out_width, GPU_ID, cudnn, r6.output_descriptor, d2, false);
 
    Conv c7(m6.out_channels, 128, 3, 1, 1, cudnn, cublas, batch_size, m6.out_width, m6.out_height, true, true, GPU_ID, m6.poolTensor, d2, false);
    Relu r7(c7.out_channels, c7.out_channels, cudnn, cublas, batch_size, c7.out_height, c7.out_width, GPU_ID, c7.output_descriptor, d2, false);
 
    Conv c8(r7.out_channels, 128, 3, 1, 1, cudnn, cublas, batch_size, r7.out_width, r7.out_height, true, true, GPU_ID, r7.output_descriptor, d2, false);
    Relu r8(c8.out_channels, c8.out_channels, cudnn, cublas, batch_size, c8.out_height, c8.out_width, GPU_ID, c8.output_descriptor, d2, false);
 
    Conv c9(r8.out_channels, final_output_depth, 3, 1, 1, cudnn, cublas, batch_size, r8.out_width, r8.out_height, true, true, GPU_ID, r8.output_descriptor, d2, false);
    Sigmoid s9(c9.out_channels, c9.out_channels, cudnn, cublas, batch_size, c9.out_height, c9.out_width, GPU_ID, c9.output_descriptor, d2, false);
       
    /* Data buffers for Forward propagation on GPU */
    // float *mask, *target; // targets for loss computation
    float *data, *c1_out, *r1_out, *m1_out, *c2_out, *r2_out, *m2_out, *c3_out, *r3_out, *c4_out, *r4_out, *c5_out, *r5_out, *m5_out;
    float *c6_out, *r6_out, *m6_out, *c7_out, *r7_out, *c8_out, *r8_out, *c9_out, *s9_out;
 
    /* Data buffers on GPU for backward propagation */
    float *grad_data, *r1_dout, *m1_dout, *r2_dout, *m2_dout, *r3_dout, *r4_dout, *r5_dout, *m5_dout;
    float *r6_dout, *m6_dout, *r7_dout, *r8_dout, *s9_dout; //conv douts not necessary as data grads are stored in their class itself

    cudaMalloc(&data, sizeof(float) * input_size);
    cudaMalloc(&grad_data, sizeof(float) * s9.output_size);
    cudaMalloc(&c1_out, sizeof(float) * c1.output_size);
    cudaMalloc(&r1_out, sizeof(float) * r1.output_size);
    cudaMalloc(&m1_out, sizeof(float) * m1.output_size);
 
        cudaMalloc(&r1_dout, sizeof(float) * r1.input_size);
        cudaMalloc(&m1_dout, sizeof(float) * m1.input_size);
 
    cudaMalloc(&c2_out, sizeof(float) * c2.output_size);
    cudaMalloc(&r2_out, sizeof(float) * r2.output_size);
    cudaMalloc(&m2_out, sizeof(float) * m2.output_size);
 
        cudaMalloc(&r2_dout, sizeof(float) * r2.input_size);
        cudaMalloc(&m2_dout, sizeof(float) * m2.input_size);
 
    cudaMalloc(&c3_out, sizeof(float) * c3.output_size);
    cudaMalloc(&r3_out, sizeof(float) * r3.output_size);
 
        cudaMalloc(&r3_dout, sizeof(float) * r3.input_size);
 
    cudaMalloc(&c4_out, sizeof(float) * c4.output_size);
    cudaMalloc(&r4_out, sizeof(float) * r4.output_size);

        cudaMalloc(&r4_dout, sizeof(float) * r4.input_size);
 
    cudaMalloc(&c5_out, sizeof(float) * c5.output_size);
    cudaMalloc(&r5_out, sizeof(float) * r5.output_size);
    cudaMalloc(&m5_out, sizeof(float) * m5.output_size);
 
        cudaMalloc(&r5_dout, sizeof(float) * r5.input_size);
        cudaMalloc(&m5_dout, sizeof(float) * m5.input_size);
 
    cudaMalloc(&c6_out, sizeof(float) * c6.output_size);
    cudaMalloc(&r6_out, sizeof(float) * r6.output_size);
    cudaMalloc(&m6_out, sizeof(float) * m6.output_size);
 
        cudaMalloc(&r6_dout, sizeof(float) * r6.input_size);
        cudaMalloc(&m6_dout, sizeof(float) * m6.input_size);
 
    cudaMalloc(&c7_out, sizeof(float) * c7.output_size);
    cudaMalloc(&r7_out, sizeof(float) * r7.output_size);
 
        cudaMalloc(&r7_dout, sizeof(float) * r7.input_size);
 
    cudaMalloc(&c8_out, sizeof(float) * c8.output_size);
    cudaMalloc(&r8_out, sizeof(float) * r8.output_size);
 
        cudaMalloc(&r8_dout, sizeof(float) * r8.input_size);
 
    cudaMalloc(&c9_out, sizeof(float) * c9.output_size);
    cudaMalloc(&s9_out, sizeof(float) * s9.output_size);
    cudaMalloc(&s9_dout, sizeof(float) * s9.input_size);
 
    // printf("Output m1 shape = %d, %d, %d, %d\n", batch_size, m1.out_channels, m1.out_height, m1.out_width);
    // printf("Output m2 shape = %d, %d, %d, %d\n", batch_size, m2.out_channels, m2.out_height, m2.out_width);
    // printf("Output m5 shape = %d, %d, %d, %d\n", batch_size, m5.out_channels, m5.out_height, m5.out_width);
    // printf("Output c6 shape = %d, %d, %d, %d\n", batch_size, c6.out_channels, c6.out_height, c6.out_width);
    // printf("Output m6 shape = %d, %d, %d, %d\n", batch_size, m6.out_channels, m6.out_height, m6.out_width);
    // printf("Output c9 shape = %d, %d, %d, %d\n", batch_size, c9.out_channels, c9.out_height, c9.out_width);
    
    /* Load Weights */
    string save_path = "../weights/";
    c9.load_params(str_to_char_arr(save_path + + "9"));
    c8.load_params(str_to_char_arr(save_path + + "8"));
    c7.load_params(str_to_char_arr(save_path + + "7"));
    c6.load_params(str_to_char_arr(save_path + + "6"));
    c5.load_params(str_to_char_arr(save_path + + "5"));
    c4.load_params(str_to_char_arr(save_path + + "4"));
    c3.load_params(str_to_char_arr(save_path + + "3"));
    c2.load_params(str_to_char_arr(save_path + + "2"));
    c1.load_params(str_to_char_arr(save_path + + "1"));

    /**** Training Loop will start from here ****/
    for(int iter = 0;iter < ITERS;iter++) {
    printf("Iteration : %d\n", iter);
    /* Final Input Output on CPU */
    // Filenames
    int image_id = (iter%num_images) + 1;
    string im_id = to_string(image_id);
    string filename = "../eval/images/resized_" + im_id + ".png";
    // float *cpu_data = (float *)malloc(sizeof(float) * input_size);
    // std::cout << "\n" << filename << "\n" << maskname << "\n" << targetname << "\n\n";
    float *cpu_data = get_img(filename); // Input image

    // for(int i=0; i<input_size; i++)
    //     cpu_data[i] = 255.0;
    float *cpu_out = (float *)malloc(sizeof(float) * c9.output_size);
    
    // float *grad_data_cpu = (float *)malloc(sizeof(float) * s9.output_size); //Upstream derivative of NxCx13x13 from Loss calculations
    // for(int i=0; i<s9.output_size; i++)
    //     grad_data_cpu[i] = 100.0;

    /* Forward Propagation */
    checkCudaErrors(cudaMemcpyAsync(data, cpu_data, sizeof(float) * input_size,  cudaMemcpyHostToDevice));
    c1.forward(data, c1_out);
    r1.forward(c1_out, r1_out);
    m1.forward(r1_out, m1_out);
    c2.forward(m1_out, c2_out);
    r2.forward(c2_out, r2_out);
    m2.forward(r2_out, m2_out);
 
    c3.forward(m2_out, c3_out);
    r3.forward(c3_out, r3_out);
    c4.forward(r3_out, c4_out);
    r4.forward(c4_out, r4_out);
 
    c5.forward(r4_out, c5_out);
    r5.forward(c5_out, r5_out);
    m5.forward(r5_out, m5_out);
    c6.forward(m5_out, c6_out);
    r6.forward(c6_out, r6_out);
    m6.forward(r6_out, m6_out);
    c7.forward(m6_out, c7_out);
    r7.forward(c7_out, r7_out);
    c8.forward(r7_out, c8_out);
    r8.forward(c8_out, r8_out);
    c9.forward(r8_out, c9_out);
    s9.forward(c9_out, s9_out);
    checkCudaErrors(cudaMemcpy(cpu_out, s9_out, sizeof(float) * c9.output_size, cudaMemcpyDeviceToHost));
    
    // Write cpu_out to output file
    filename = "../eval/predictions/resized_" + im_id + ".txt";
    FILE *fp = fopen(str_to_char_arr(filename), "a");
    if (!fp) {
      printf("FILE ERROR: Cannot open file\n");
      exit(2);
    }

    printf("%d\n\n\n", c9.output_size);
    for(int k = 0;k < c9.output_size;k++) {
        fprintf(fp, "%f\n", cpu_out[k]);
    }

    // fwrite(&cpu_out[0], sizeof(float), s9.output_size, fp);
    fclose(fp);

    free(cpu_data);
    }   

    printf("Done\n\n\n");
    checkCudaErrors(cublasDestroy(cublas));
    checkCUDNN(cudnnDestroy(cudnn));

    return 0;
}
