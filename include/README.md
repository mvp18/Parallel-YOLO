## Class constructors, forward, and backward syntax

 - Convolution 2D
 ```
 Conv(int _in_channels, int _out_channels, int _kernel_size, int padding, int stride,
      cudnnHandle_t _cudnn, cublasHandle_t _cublas, int batch_size, int width, int height,
      bool use_backward_filter, bool use_backward_data, int gpu_id,
      cudnnTensorDescriptor_t& _input_descriptor, cudnnTensorDescriptor_t& _output_descriptor,
      int init_io_desc)
 
 void forward(float *d_input, float *d_output);
 
 void backward(float *data_grad_above, cudnnTensorDescriptor_t &input_descriptor, float *data_below)
 ```
 - MaxPool 2D
 ```
 MaxPoolLayer(int size, int stride, int padding, int batch_size, int conv_out_channel,
              int conv_out_height, int conv_out_width, int _gpu_id, cudnnHandle_t &_cudnnHandle,
              cudnnTensorDescriptor_t& _input_descriptor, cudnnTensorDescriptor_t& _output_descriptor,
              int init_io_desc)
 
 void forward(float* input_tensor, float* output_tensor);
 
 void backward(float* input_tensor, float *input_gradient, float *output_tensor, float* output_gradient);
 ```
 - Relu
 ```
 Relu(int _in_channels, int _out_channels, cudnnHandle_t _cudnn, cublasHandle_t _cublas,
      int batch_size, int height, int width, int _gpu_id,
       cudnnTensorDescriptor_t& _input_descriptor, cudnnTensorDescriptor_t& _output_descriptor,
       int init_io_desc)
 
 void forward(float *d_input, float *d_output);
 
 void backward(float *grad_above, float *grad_out);
 ```

 - Sigmoid
 ```
 Sigmoid(int _in_channels, int _out_channels, cudnnHandle_t _cudnn, cublasHandle_t _cublas, int batch_size, int height, int width, int _gpu_id,
             cudnnTensorDescriptor_t& _input_descriptor, cudnnTensorDescriptor_t& _output_descriptor, int init_io_desc);
 
 void forward(float *d_input, float *d_output);

 void backward(float *grad_above, float *grad_out);

 ```

 - Softmax
 ```
 Softmax(int _in_channels, int _out_channels, cudnnHandle_t _cudnn, cublasHandle_t _cublas,
             int batch_size, int height, int width, int _gpu_id);

 void forward(float *d_input, float *d_output);

 void backward(float *grad_above, float *grad_out);

 ```

 
- data_utils
```
char* str_to_char_arr(string str);

float* get_img(string filename_);

float* get_float_array(string filename_);
```

- mse
```
__global__ void SoftmaxLossBackprop(const float *label, int num_labels, int batch_size, float *diff);
```
 
