# Parallel-YOLO

Objective
=========

Vision based perception systems are critical for autonomous-driving vehicle products. The objective of this project is to train a tiny-YOLO based CNN pipeline that can detect objects on composite images derived from multiple camera feeds on a vehicle. 


Instructions
============
To compile with and run with CMake, run the following commands:

```bash
bash main.sh
cd build
```

This will create targets in a C++ readable format, create evaluation targets and compile the code.

To train the model, modify the configuration parameters in `train.cu`
```
input_height = input_width = 416;
in_channels = 3;
batch_size = 1;
num_classes = 15;
num_anchors = 5;
learning_rate = -0.001;
num_images = 12;
epochs = 10000;
ITERS = epochs * num_images;
SAVE_FREQUENCY = 50;
```

Now run the following
```bash
cd build
./train
```

To evaluate the model, place the evaluation images in `$ROOT/eval/images/`, create an empty directory as `$ROOT/eval/predictions` and modify the configuration parameters in `test.cu`
```
input_height = input_width = 416;
in_channels = 3;
batch_size = 1;
num_classes = 1;
num_anchors = 5;
num_images = 1;
```

Now run the following from `$ROOT/build`
```bash
./eval
```

To see predictions on the terminal, run the following from `$ROOT/scripts/`
```bash
python infer_targets.py
```

The resources at hand currently are insufficient for training this architecture. So, the evaluation file may not give expected results.

If compiling under linux, make sure to either set the CUDNN_PATH environment variable to the path CUDNN is installed to, or extract CUDNN to the CUDA toolkit path.

To enable gflags support, uncomment the line in CMakeLists.txt. In the Visual Studio project, define the macro USE_GFLAGS.

Reference Paper
==============

Re-Thinking CNN Frameworks for Time-Sensitive Autonomous-Driving Applications: Addressing an Industrial Challenge
