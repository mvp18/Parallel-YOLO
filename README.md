# Parallel-YOLO

Task Left
=========

- Build a small model and train on Mnist (for testing purposes)
- Build the tiny-yolo architecture
- And much more...


Train Model
==========
To compile with and run with CMake, run the following commands:

```bash
bash create_standard_data.sh
mkdir build
cd build
cmake ..
make
./train
```

If compiling under linux, make sure to either set the ```CUDNN_PATH``` environment variable to the path CUDNN is installed to, or extract CUDNN to the CUDA toolkit path.

To enable gflags support, uncomment the line in CMakeLists.txt. In the Visual Studio project, define the macro ```USE_GFLAGS```.
