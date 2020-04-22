rm -rf weights
mkdir weights
rm -rf build
rm -rf data
rm -rf eval
virtualenv env
source env/bin/activate
pip install -r requirements.txt
cd processing
python resize.py
bash get_standard_annotations.sh
cd ../scripts
python create_targets.py standard
deactivate
cd ..
mkdir weights
mkdir build
cd build
cmake ..
make

