## YOLO architecture implemented
``` 
    //kernel = 13, padding = 1, stride = 2
    Conv c1(in_channels, 192, 13, 1, 2, cudnn, cublas, batch_size, input_width, input_height, true, false, GPU_ID, d1, d2, true);
    Relu r1(c1.out_channels, c1.out_channels, cudnn, cublas, batch_size, c1.out_height, c1.out_width, GPU_ID, c1.output_descriptor, d2, false);
    //size = 2, stride = 2, padding = 0
    MaxPoolLayer m1(2, 2, 0, batch_size, r1.out_channels, r1.out_height, r1.out_width, GPU_ID, cudnn, r1.output_descriptor, d2, false);
 
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
```
