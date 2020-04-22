class Relu
{
    // ReLu Layer class

    public:
        // Declare public variables
        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        cudnnTensorDescriptor_t input_descriptor;
        cudnnTensorDescriptor_t output_descriptor;

        cudnnActivationDescriptor_t activation_descriptor;
        
        cudnnHandle_t cudnn;
        cublasHandle_t cublas;
 
        // These variables will be on GPU as cache for backward pass
        float *din;  //Input to ReLU layer
        float *dot;  //Output of ReLU layer
 
        // These variables will be on CPU 
        int input_size, output_size;
        int input_height, out_height;
        int in_channels, out_channels;
        int input_width, out_width;
        int gpu_id;
        float *din_cpu; //Cache for backprop
        float *dot_cpu; //Cache for backprop
 
        Relu(int _in_channels, int _out_channels, cudnnHandle_t _cudnn, cublasHandle_t _cublas, int batch_size, int height, int width, int _gpu_id,
             cudnnTensorDescriptor_t& _input_descriptor, cudnnTensorDescriptor_t& _output_descriptor, int init_io_desc)
        {
            cudnn = _cudnn;
            cublas = _cublas;
            gpu_id = _gpu_id;

            checkCudaErrors(cudaSetDevice(gpu_id));
         
            in_channels = _in_channels;
            out_channels = _out_channels;
            input_width = out_width = width;
            out_height = height;
            input_height = height;
         
            if(init_io_desc)
            {
                checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
                checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor, 
                                                          CUDNN_TENSOR_NCHW,
                                                          CUDNN_DATA_FLOAT,
                                                          batch_size,
                                                          in_channels,
                                                          height,
                                                          width));
            }
            else
                input_descriptor = _input_descriptor;


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
        
        // Destructor for de-allocating memory
        ~Relu()
        {
            checkCUDNN(cudnnDestroyTensorDescriptor(input_descriptor));
            checkCUDNN(cudnnDestroyTensorDescriptor(output_descriptor));
            checkCUDNN(cudnnDestroyActivationDescriptor(activation_descriptor));
        }
 
        void forward(float *d_input, float *d_output)
        {
            // Performs forward pass for relu layer
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
 
        void backward(float *grad_above, float *grad_out) //
        {
            // Performs backward pass for relu layer
            checkCUDNN(cudnnActivationBackward(
                cudnn,
                activation_descriptor,
                &alpha,
                output_descriptor,
                dot,
                output_descriptor,
                grad_above,
                input_descriptor,
                din,
                &beta,
                input_descriptor,
                grad_out
            ));
        }
};