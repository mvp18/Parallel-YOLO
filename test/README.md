## Tests

- **test_convolution()**
  - Input data
  ```
  const float data_[5][4] = {{1, 6, 11, 16},
                               {2, 7, 12, 17},
                               {3, 8, 13, 18},
                               {4, 9, 14, 19},
                               {5, 10, 15, 20}};
  ```
  - Upstream derivatives for backprop
  ```                             
  const float data_grad[5][4] = {{0, 0, 0, 0},
                                    {0, 10, 10, 0},
                                    {0, 0, 0, 0},
                                    {0, 10, 10, 0},
                                    {0, 0, 0, 0}};
  ```
  - Kernel
  ```
  const float kernel_template[3][3] = {
          {1.0/9.0, 1.0/9.0, -1.0/9.0},
          {1.0/9.0, -1.0/9.0, 1.0/9.0},
          {1.0/9.0, 1.0/9.0, 1.0/9.0}
        };
  ```

- **test_relu()**
  - Performs only Relu forward and backward with data defined inside function.
  
- **test_mpl()**
  - Conv followed by maxpool test. Data same as conv. Only the `data_grad` has all elements = 10.
- **test_conv_relu_maxpool()**
  - Conv - > Relu -> Maxpool test. Same data as above.
