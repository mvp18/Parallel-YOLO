# The script does the following:

The initial data file has the following format:

```
PASCAL Annotation Version 1.00

Image filename : "VOC2006/PNGImages/000004.png"

Image size (X x Y x C) : 335 x 500 x 3

Database : "The VOC2006 Database"

Objects with ground truth : 3 { "PAScat" "PAScat" "PAScat" }


 Note that there might be other objects in the image
 
 for which ground truth data has not been provided.

 Top left pixel co-ordinates : (1, 1)


 Details for object 1 ("PAScat")
 
Original label for object 1 "PAScat" : "PAScat"

Bounding box for object 1 "PAScat" (Xmin, Ymin) - (Xmax, Ymax) : (161, 185) - (200, 241)


 Details for object 2 ("PAScat")
 
Original label for object 2 "PAScat" : "PAScat"

Bounding box for object 2 "PAScat" (Xmin, Ymin) - (Xmax, Ymax) : (108, 192) - (150, 230)


 Details for object 3 ("PAScat")
 
Original label for object 3 "PAScat" : "PAScat"

Bounding box for object 3 "PAScat" (Xmin, Ymin) - (Xmax, Ymax) : (100, 256) - (175, 380)
```


Which finally gets converted to:
```  
3

PAScat 199 153 249 201

PAScat 134 159 187 192

PAScat 124 212 218 317
```

The first element specifying the number of objects(n), followed by n lines to describe each object. The content of each line is as follows:
```
ObjectName xmin ymin xmax ymax.
```
These version of `xmin`, `xmax`, `ymin`, `ymax` are scaled. The lower coordinates are rounded down and upper coordinates are rounded up.
