rm -r build
mkdir build
cd build
cmake ..
make
./run_test
cd ..
rm -r build
