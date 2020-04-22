class MaxPoolLayer
{
    // Max Pool Layer class
    public:
        // Declare public variables
        float alpha = 1.0f, beta = 0.0f;
        int gpu_id, input_height, input_width, input_size, out_height, out_width, output_size, out_channels;

        cudnnPoolingDescriptor_t poolDesc;
        cudnnTensorDescriptor_t input_descriptor, poolTensor;

        cudnnHandle_t cudnnHandle;

        MaxPoolLayer(int size, int stride, int padding, int batch_size, int conv_out_channel, int conv_out_height, int conv_out_width, int _gpu_id, cudnnHandle_t &_cudnnHandle,
                    cudnnTensorDescriptor_t& _input_descriptor, cudnnTensorDescriptor_t& _output_descriptor, int init_io_desc)
        {

            // Assign Handles
            cudnnHandle=_cudnnHandle;
        
            // Assign the GPU id to run on
            gpu_id = _gpu_id;
            // checkCudaErrors(cudaSetDevice(gpu_id));

            // Forward Propagation Descriptors
            input_width = conv_out_width;
            input_height = conv_out_height;
            input_size = batch_size * conv_out_channel * input_height * input_width;

            out_height = (conv_out_height - size + 2*padding) / stride + 1;
            out_width = (conv_out_width - size + 2*padding) / stride + 1;
            out_channels = conv_out_channel;

            output_size = batch_size * conv_out_channel* out_height * out_width;

            // Input Tensor (it is the output tensor from the convolution layer)
            if(init_io_desc)
            {
                checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
                checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,                 // Uses Tensor Descriptor
                                                      CUDNN_TENSOR_NCHW,                //format
                                                      CUDNN_DATA_FLOAT,                 //dataType
                                                      batch_size,                       //batch_size
                                                      conv_out_channel,                 //channels
                                                      conv_out_height,                  //image_height
                                                      conv_out_width));                 //image_width
            }
            else
                input_descriptor = _input_descriptor;

            
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
                                                out_height,
                                                out_width));

            
        }

        // Destructor for de-allocating memory
        ~MaxPoolLayer() 
        {   
            // checkCudaErrors(cudaSetDevice(gpu_id));
            /* Note : Do not delete cublas and cudnn here and destroy them where the final model will be defined */
            checkCUDNN(cudnnDestroyTensorDescriptor(input_descriptor));
            checkCUDNN(cudnnDestroyTensorDescriptor(poolTensor));
            checkCUDNN(cudnnDestroyPoolingDescriptor(poolDesc));
        }

        void forward(float* input_tensor, float* output_tensor/*, cudnnTensorDescriptor_t& _input_descriptor*/)
        {
            //Performs forward pass for convolution layer
            checkCUDNN(cudnnPoolingForward(cudnnHandle,         //handle
                                          poolDesc,            //poolingDesc
                                          &alpha,              //alpha
                                          input_descriptor,    //xDesc
                                          input_tensor,        //x
                                          &beta,               //beta
                                          poolTensor,          //yDesc
                                          output_tensor));     //y    
        }

        void backward(float* input_tensor, float *input_gradient, float *output_tensor, float* output_gradient/*, cudnnTensorDescriptor_t& _input_descriptor*/)
        {
          /*Performs backward pass for convolution layer
            conv + relu layer 1-> max pool -> conv layer 2
            input_tensor -> input from conv1
            input_gradient -> gradient from conv 2
            output_tensor -> output of max pool layer
            output_gradient -> output gradient  
          */
        
            checkCUDNN(cudnnPoolingBackward(cudnnHandle,              //handle
                                            poolDesc,                 //poolingDesc
                                            &alpha,                   //alpha
                                            poolTensor,               //yDesc
                                            output_tensor,            //Output of max pool layer
                                            poolTensor,               //dyDesc
                                            input_gradient,           //The gradient recieved from conv_layer_2
                                            input_descriptor,         //xDesc
                                            input_tensor,             //Input from conv_layer_1
                                            &beta,                    //beta
                                            input_descriptor,         //dxDesc
                                            output_gradient));        //Output differential tensor 

        }
        
        void update_weights()
        {
            // No weights to updates :)
            return;
        }
};