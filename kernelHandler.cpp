#include "kernelHandler.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

using namespace std;
using namespace cv;

kernelHandler::kernelHandler(const char *kernelFilename)
{
    this->kernelFilename = kernelFilename;
    FILE *kernelFile = fopen(kernelFilename, "r");
    if (kernelFile == NULL)
    {
        perror("Error in opening kernel file");
    }

    loadAllKernels();
}

void kernelHandler::loadAllKernels()
{
    std::ifstream file(kernelFilename);
    std::string str;

    std::getline(file, str);
    numOfKernels = stoi(str);

    for (int i = 0; i < numOfKernels; i++)
    {
        std::getline(file, str);
        int kernel_dim = stoi(str);

        kernel kl;
        kl.dimension = kernel_dim;
        kl.matrix = new float *[kernel_dim];

        for (int j = 0; j < kernel_dim; j++)
        {
            kl.matrix[j] = new float[kernel_dim];
            std::getline(file, str);
            loadRow(kl.matrix[j], kernel_dim, str);
        }
        kernels.push_back(kl);
    }
}

void kernelHandler::loadRow(float *row, int kernelDimension, string buf)
{
    std::stringstream ss(buf);
    for (int i = 0; i < kernelDimension; i++)
    {
        string substr;
        getline(ss, substr, ',');
        row[i] = std::stof(substr);
    }
}

void kernelHandler::printKernel()
{
    printf("\nNum of kernels : %d", numOfKernels);
    for (int k = 0; k < numOfKernels; k++)
    {
        printf("\n%dx%d", kernels[k].dimension, kernels[k].dimension);
        for (int i = 0; i < kernels[k].dimension; i++)
        {
            printf("\n");
            for (int j = 0; j < kernels[k].dimension; j++)
            {
                printf("%f ", kernels[k].matrix[i][j]);
            }
        }
        std::cout << "\n====================================";
    }
}

int kernelHandler::getNumOfKernels()
{
    return numOfKernels;
}

kernel kernelHandler::getKernel(int index)
{
    return kernels[index];
}

std::vector<kernel> kernelHandler::getKernels()
{
    return kernels;
}

cv::Mat kernelHandler::returnMatrix(int index)
{
    Mat mat = Mat::ones(kernels[index].dimension, kernels[index].dimension, CV_32F);

    for (int i = 0; i < kernels[index].dimension; i++)
    {
        for (int j = 0; j <kernels[index].dimension; j++)
        {
            mat.at<float>(i,j) = kernels[index].matrix[i][j];
        }
    }
    return mat;
}