#!/bin/bash


i=1
cwd=$(pwd)
rm -rf $cwd/../data/standard/Annotations
mkdir -p $cwd/../data/standard/Annotations
for file in ../sample/labels/*
do

    x=$( grep "Image size" $file | cut -d "x" -f3 | grep -Eo -m1 '[[:digit:]]*' )
    y=$( grep "Image size" $file | cut -d "x" -f4 | grep -Eo -m1 '[[:digit:]]*' )

    objectnum=$( grep "Objects with ground truth" $file | grep -Eo -m1 '[0-9]{1,4}' )

    grep "Bounding box for object " $file | cut -d "\"" -f2 > tmp1 
    grep "Bounding box for object " $file | cut -d ":" -f2 | cut -d "(" -f2 | cut -d "," -f1 | grep -Eo  '[[:digit:]]*' > tmp2
    grep "Bounding box for object " $file | cut -d ":" -f2 | cut -d "(" -f2 | cut -d "," -f2 | cut -d ")" -f1 | grep -Eo '[[:digit:]]*' > tmp3
    grep "Bounding box for object " $file | cut -d ":" -f2 | cut -d "(" -f3 | cut -d "," -f1 | grep -Eo '[[:digit:]]*' > tmp4
    grep "Bounding box for object " $file | cut -d ":" -f2 | cut -d "(" -f3 | cut -d "," -f2 | cut -d ")" -f1 | grep -Eo '[[:digit:]]*' > tmp5


    awk -v SF1="$x" -F " "   '{$1=int(($1*416)/(SF1));print}' tmp2 > tmp6
    awk -v SF2="$y" -F " "   '{$1=int(($1*416)/(SF2));print}' tmp3 > tmp7
    awk -v SF3="$x" -F " "   '{$1=int(($1*416 + SF3 - 1 )/(SF3));print}' tmp4 > tmp8
    awk -v SF4="$y" -F " "   '{$1=int(($1*416 + SF4 - 1 )/(SF4));print}' tmp5 > tmp9

    ( [ -s tmp1 ] && [ -s tmp2 ] && [ -s tmp3 ] && [ -s tmp4 ] && [ -s tmp5 ] )   || echo "Error in $i case";
    ( [ -s tmp1 ] && [ -s tmp2 ] && [ -s tmp3 ] && [ -s tmp4 ] && [ -s tmp5 ] )   || break;

    echo $objectnum > $cwd/../data/standard/Annotations/resized_$i.txt
    paste -d " " tmp1 tmp6 tmp7 tmp8 tmp9  >> $cwd/../data/standard/Annotations/resized_$i.txt
    echo "Done $i files"
    i=$(($i+1))
    # if [ $i -eq 100 ]; then
	# 	break
	# fi

done


rm tmp*
