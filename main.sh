rm -rf weights
mkdir weights
rm -rf build
rm -rf data
rm -rf eval
cd processing
python resize.py
bash get_standard_annotations.sh
cd ../scripts
python create_targets.py standard
cd ..
mkdir weights
mkdir build
cd build
cmake ..
make

