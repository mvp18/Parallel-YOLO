# Parallel-YOLO

Task Left
=========

- Build a small model and train on Mnist (for testing purposes)
- Build the tiny-yolo architecture
- And much more...

Run Tests
==========
To ensure all layers are compiled successfully and execute successfully, run the command below from the root directory.

```bash
bash run_tests.sh
```

Train Model
==========

```bash
bash create_standard_data.sh
mkdir build
cd build
cmake ..
make
./train
```

Compilation
===========

The project can either be compiled with CMake (cross-platform) or Visual Studio.

To compile with CMake, run the following commands:
```bash
~: $ cd Parallel-YOLO/
~/Parallel-YOLO: $ mkdir build
~/Parallel-YOLO: $ cd build/
~/Parallel-YOLO/build: $ cmake ..
~/Parallel-YOLO/build: $ make

Run simply by running the required executable. eg- max_pool
~/Parallel-YOLO/build: $ ./max_pool
```

If compiling under linux, make sure to either set the ```CUDNN_PATH``` environment variable to the path CUDNN is installed to, or extract CUDNN to the CUDA toolkit path.

To enable gflags support, uncomment the line in CMakeLists.txt. In the Visual Studio project, define the macro ```USE_GFLAGS```.
