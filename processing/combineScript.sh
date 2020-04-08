#!/bin/bash


i=1
j=1
totobject=0
objectnum=0
rm -rf NewAnnotations
mkdir -p NewAnnotations
for file in Annotations/*
do

    x=$( grep "Image size" $file | cut -d "x" -f3 | grep -Eo -m1 '[[:digit:]]*' )
    y=$( grep "Image size" $file | cut -d "x" -f4 | grep -Eo -m1 '[[:digit:]]*' )

    objectnum=$( grep "Objects with ground truth" $file | grep -Eo -m1 '[0-9]{1,4}' )
    grep "Bounding box for object " $file | cut -d "\"" -f2 > tmp1 
    grep "Bounding box for object " $file | cut -d ":" -f2 | cut -d "(" -f2 | cut -d "," -f1 | grep -Eo  '[[:digit:]]*' > tmp2
    grep "Bounding box for object " $file | cut -d ":" -f2 | cut -d "(" -f2 | cut -d "," -f2 | cut -d ")" -f1 | grep -Eo '[[:digit:]]*' > tmp3
    grep "Bounding box for object " $file | cut -d ":" -f2 | cut -d "(" -f3 | cut -d "," -f1 | grep -Eo '[[:digit:]]*' > tmp4
    grep "Bounding box for object " $file | cut -d ":" -f2 | cut -d "(" -f3 | cut -d "," -f2 | cut -d ")" -f1 | grep -Eo '[[:digit:]]*' > tmp5

    if [[ $j -eq 1 ]]; then # Image 1
        totobject=$(($totobject+$objectnum))
        awk -v SF1="$x" -F " "   '{$1=int(($1*208)/(SF1));print}' tmp2 > tmp6
        awk -v SF2="$y" -F " "   '{$1=int(($1*208)/(SF2));print}' tmp3 > tmp7
        awk -v SF3="$x" -F " "   '{$1=int(($1*208 + SF3 - 1 )/(SF3));print}' tmp4 > tmp8
        awk -v SF4="$y" -F " "   '{$1=int(($1*208 + SF4 - 1 )/(SF4));print}' tmp5 > tmp9

        ( [ -s tmp1 ] && [ -s tmp2 ] && [ -s tmp3 ] && [ -s tmp4 ] && [ -s tmp5 ] )   || echo "Error in $i'th file";
        ( [ -s tmp1 ] && [ -s tmp2 ] && [ -s tmp3 ] && [ -s tmp4 ] && [ -s tmp5 ] )   || break;

        paste -d " " tmp1 tmp6 tmp7 tmp8 tmp9  >> NewAnnotations/resized_$i.txt
        # echo "Doing $i'th file"
    elif [[ $j -eq 2 ]]; then # Image 2
        totobject=$(($totobject+$objectnum))
        awk -v SF1="$x" -F " "   '{$1=int(($1*208)/(SF1)) + 208;print}' tmp2 > tmp6
        awk -v SF2="$y" -F " "   '{$1=int(($1*208)/(SF2));print}' tmp3 > tmp7
        awk -v SF3="$x" -F " "   '{$1=int(($1*208 + SF3 - 1 )/(SF3)) + 208;print}' tmp4 > tmp8
        awk -v SF4="$y" -F " "   '{$1=int(($1*208 + SF4 - 1 )/(SF4));print}' tmp5 > tmp9

        ( [ -s tmp1 ] && [ -s tmp2 ] && [ -s tmp3 ] && [ -s tmp4 ] && [ -s tmp5 ] )   || echo "Error in $i'th file";
        ( [ -s tmp1 ] && [ -s tmp2 ] && [ -s tmp3 ] && [ -s tmp4 ] && [ -s tmp5 ] )   || break;

        paste -d " " tmp1 tmp6 tmp7 tmp8 tmp9  >> NewAnnotations/resized_$i.txt
        # echo "Doing $i'th file"
    elif [[ $j -eq 3 ]]; then # Image 3

        totobject=$(($totobject+$objectnum))
        awk -v SF1="$x" -F " "   '{$1=int(($1*208)/(SF1));print}' tmp2 > tmp6
        awk -v SF2="$y" -F " "   '{$1=int(($1*208)/(SF2))+208;print}' tmp3 > tmp7
        awk -v SF3="$x" -F " "   '{$1=int(($1*208 + SF3 - 1 )/(SF3));print}' tmp4 > tmp8
        awk -v SF4="$y" -F " "   '{$1=int(($1*208 + SF4 - 1 )/(SF4))+208;print}' tmp5 > tmp9

        ( [ -s tmp1 ] && [ -s tmp2 ] && [ -s tmp3 ] && [ -s tmp4 ] && [ -s tmp5 ] )   || echo "Error in $i'th file";
        ( [ -s tmp1 ] && [ -s tmp2 ] && [ -s tmp3 ] && [ -s tmp4 ] && [ -s tmp5 ] )   || break;

        paste -d " " tmp1 tmp6 tmp7 tmp8 tmp9  >> NewAnnotations/resized_$i.txt
        # echo "Doing $i'th file"
    else # Image 4
        totobject=$(($totobject+$objectnum))
        awk -v SF1="$x" -F " "   '{$1=int(($1*208)/(SF1))+208;print}' tmp2 > tmp6
        awk -v SF2="$y" -F " "   '{$1=int(($1*208)/(SF2))+208;print}' tmp3 > tmp7
        awk -v SF3="$x" -F " "   '{$1=int(($1*208 + SF3 - 1 )/(SF3))+208;print}' tmp4 > tmp8
        awk -v SF4="$y" -F " "   '{$1=int(($1*208 + SF4 - 1 )/(SF4))+208;print}' tmp5 > tmp9

        ( [ -s tmp1 ] && [ -s tmp2 ] && [ -s tmp3 ] && [ -s tmp4 ] && [ -s tmp5 ] )   || echo "Error in $i'th file";
        ( [ -s tmp1 ] && [ -s tmp2 ] && [ -s tmp3 ] && [ -s tmp4 ] && [ -s tmp5 ] )   || break;

        paste -d " " tmp1 tmp6 tmp7 tmp8 tmp9  >> NewAnnotations/resized_$i.txt
        sed -i "1i $totobject" NewAnnotations/resized_$i.txt
        echo "Done $i'th file"
        j=0
        i=$(($i+1))
        totobject=0
    fi
    
    j=$(($j+1))
 #    if [ $i -eq 20 ]; then
	# 	break
	# fi

done


rm tmp*
