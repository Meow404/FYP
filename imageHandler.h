#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>

class imageHandler{
    char *imagePath;
    float *image_array;
    unsigned int width, height;

public:
    imageHandler(const char* filepath);
    imageHandler(int width, int height, float* image_array);

    float* get1DImageArray();
    float** get2DImageArray();

    void print1DImage();
    void print2DImage();

    int getHeight();
    int getWidth();

    void saveImage(string filepath);
};
