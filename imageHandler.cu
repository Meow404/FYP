#include "imageHandler.h"
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>

imageHandler::imageHandler(const char* filepath){
    imagePath = sdkFindFilePath(filepath, NULL);

    if (imagePath == NULL)
    {
        printf("Unable to source image file: %s\n", filepath);
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM(imagePath, &image_array, &width, &height);
}

imageHandler::imageHandler(int width, int height, float* image_array){
    this->width = width;
    this->height = height;
    this->image_array = image_array;
}

float* imageHandler::get1DImageArray(){
    return image_array;
}

float** imageHandler::get2DImageArray(){
    float **new_image_array = new float*[height];
    for(int i = 0; i < height; ++i) {
        new_image_array[i] = new float[width];
}
    
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            new_image_array[y][x] = image_array[y * width + x];

    return new_image_array;
}

void imageHandler::print1DImage(){
    for (int y = 0; y < height; y++)
    {   std::cout << "|";
        for (int x = 0; x < width; x++)
            std::cout << image_array[y * width + x] << "|";
        std::cout << "\n";
    }
}

void imageHandler::print2DImage(){
    float **new_image = get2DImageArray();
    for (int y = 0; y < height; y++)
    {   std::cout << "|";
        for (int x = 0; x < width; x++)
            std::cout << new_image[y][x] << "|";
        std::cout << "\n";
    }
}

int imageHandler::getHeight(){
    return height
}

int imageHandler::getWidth(){
    return width
}

void imageHandler::saveImage(string filepath){
    sdkSavePGM(filepath, image_array, width, height);
}
