char* str_to_char_arr(string str) {
	char *c = strcpy(new char[str.length() + 1], str.c_str());
	return c;
}

/* Great resource for compilation : https://stackoverflow.com/questions/30980383/cmake-compile-options-for-libpng */
float* get_img(string filename_)
{	
	int width, height;
	png_byte color_type;
	float *image;
	png_byte bit_depth;
	png_bytep *row_pointers = NULL;

	char *filename = str_to_char_arr(filename_);
	FILE *fp = fopen(filename, "rb");
	png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if(!png) abort();
	png_infop info = png_create_info_struct(png);
	if(!info) abort();
	if(setjmp(png_jmpbuf(png))) abort();
	png_init_io(png, fp);
	png_read_info(png, info);
	width      = png_get_image_width(png, info);
	height     = png_get_image_height(png, info);
	color_type = png_get_color_type(png, info);
	bit_depth  = png_get_bit_depth(png, info);
	if(bit_depth == 16) png_set_strip_16(png);
	if(color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
	if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png);
	if(png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
	if(color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_PALETTE) png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
	if(color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA) png_set_gray_to_rgb(png);
	png_read_update_info(png, info);
	if (row_pointers) abort();
	row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
	for(int y = 0; y < height; y++) row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png,info));
	png_read_image(png, row_pointers);
	fclose(fp);
	png_destroy_read_struct(&png, &info, NULL);

	int x,y;

	image = (float *)malloc(3 * height * width * sizeof(int*));
	int stride = height * width;
	for(y = 0; y < height; y++)
	{
		png_bytep row = row_pointers[y];
		for(x = 0; x < width; x++) 
		{
			png_bytep px = &(row[x * 4]);
			image[y * width + x]=px[0];
			image[y * width + x + stride] = px[1];
			image[y * width + x + 2 * stride] = px[2];
		}
	}
	
	return image;
}

float* get_float_array(string filename_) {
	char *filename = str_to_char_arr(filename_);
	// Reads a file of floats and returns it in a pointer
	vector<float> arr;
	// read points
	FILE *in_file;
	in_file = fopen(filename, "r");

    if (in_file == NULL)
    {
        printf("Can't open file for reading.\n");
        exit(0);
    }
    
	float result;
	float n;
	float sum = 0.0;
	while(result = fscanf(in_file, "%f", &n) != EOF) {
		arr.push_back(n);
		sum += n;
		// printf("%f\n", result);
	}

	printf("%f Sum of read array, %lu\n", sum, arr.size());

	float *out = (float *)malloc(sizeof(float) * arr.size());
	for(int i = 0;i < arr.size();i++) out[i] = arr[i];
    fclose(in_file);

	return out;
}
