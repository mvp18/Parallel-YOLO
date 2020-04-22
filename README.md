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

To check the working of building blocks of the model,
```bash
./run_test
```

To train the model,
```bash
./train
```

To evaluate the model,
```bash
./eval
```

The resources at hand currently are insufficient for training this architecture. So, the evaluation file may not give expected results.

If compiling under linux, make sure to either set the CUDNN_PATH environment variable to the path CUDNN is installed to, or extract CUDNN to the CUDA toolkit path.

To enable gflags support, uncomment the line in CMakeLists.txt. In the Visual Studio project, define the macro USE_GFLAGS.

Reference Paper
==============

Re-Thinking CNN Frameworks for Time-Sensitive Autonomous-Driving Applications: Addressing an Industrial Challenge
