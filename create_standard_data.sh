cd processing
python resize.py
bash get_standard_annotations.sh
cd ../scripts
python create_targets.py standard
