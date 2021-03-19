#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include "opencv2/core/cuda.hpp"
#include <iostream>
#include <stdio.h>
#include "kernelHandler.h"

#define ITERATIONS 100

using namespace cv;
using namespace std;

int opencvConvolve(char* file_path)
{
    // Read the image file
    cv::Mat image = imread(file_path);
    cv::Mat result, temp;
    const int ratio = 3;
    const int lowThreshold = 20;
    const int kernel_size = 3;

    kernelHandler kh = kernelHandler("kernels.txt");

    // Check for failure
    if (image.empty())
    {
        cout << "Could not open or find the image" << endl;
        cin.get(); //wait for any key press
        return -1;
    }

    // cout << "image = " << endl
    //      << " " << image << endl
    //      << endl;

    cv::cvtColor(image, temp, cv::COLOR_BGR2GRAY);
    // kh.printKernel();
    for (int i = 0; i < kh.getNumOfKernels(); i++)
    {
        for (int j = 0; j < ITERATIONS; j++)
        {
            int dim = kh.getKernel(i).dimension;
            cv::Mat k = kh.returnMatrix(i);
            
            cv::normalize(k, k, 1.0, 0.0, NORM_L1);

            // Ptr<cuda::Convolution> convolver = cuda::createConvolution(k.size);
            // convolver->convolve(result, k, result);
            cv::filter2D(temp, result, -1, k, Point(-1, -1), 5.0, BORDER_REPLICATE);
        }

        char output_file[50], file_name[50];
        sprintf(file_name, "_%dx%d_opencv_serial_out.pgm", kh.getKernel(i).dimension, kh.getKernel(i).dimension);
        strcpy(output_file, file_path);
        strcpy(output_file + strlen(file_path) - 4, file_name);
        cout << "\nWriting to : " << output_file;
        cout << result;
        imwrite(output_file, result);
    }
    //  image.copyTo(result);
}

int opencvCUDAConvolve()
{
    // Read the image file
    cv::Mat image = imread("res/images/lena_bw.pgm");
    cv::Mat result;
    cv::cuda::GpuMat gpu_image, gpu_result, gpu_kernel;
    const int ratio = 3;
    const int lowThreshold = 20;
    const int kernel_size = 3;

    kernelHandler kh = kernelHandler("kernels.txt");

    // Check for failure
    if (image.empty())
    {
        cout << "Could not open or find the image" << endl;
        cin.get(); //wait for any key press
        return -1;
    }

    cout << "image = " << endl
         << " " << image << endl
         << endl;

    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    gpu_image.upload(image);
    gpu_image.convertTo(gpu_image, CV_32FC1);

    kh.printKernel();
    for (int i = 0; i < kh.getNumOfKernels(); i++)
    {
        int dim = kh.getKernel(i).dimension;
        cv::Mat k = kh.returnMatrix(i);
        cout << "kernel = " << k << endl;
        cv::normalize(k, k, 1.0, 0.0, NORM_L1);
        cout << "kernel = " << k << endl;
        gpu_kernel.upload(k);
        gpu_kernel.convertTo(gpu_kernel, CV_32FC1);
        cv::normalize(k, k, 1.0, 0.0, NORM_L1);
        cout << "kernel = " << k;

        Ptr<cuda::Convolution> convolver = cuda::createConvolution(cv::Size(dim, dim));
        convolver->convolve(gpu_image, gpu_kernel, gpu_result);
        // cv::filter2D(result, result, -1, kernel, Point(-1, -1), 5.0, BORDER_REPLICATE);
        gpu_result.download(result);
        cout << "result = " << endl
             << " " << result << endl
             << endl;

        char filename[50];
        sprintf(filename, "res/images/result_CUDA_convolve_%d_kernel.jpg", dim);
        cout << "Writing to : " << filename;
        imwrite(filename, result);
    }
    //  image.copyTo(result);
}

int main(int argc, char **argv)
{
    printf("Image convolution project \n");
    printf("Please select an option \n");
    printf("1 - Serial OpenCV Convolution Implementation \n");
    printf("2 - OpenCV CUDA Convolution implementation \n");
    // printf("3 - Shared memory implementation \n");
    // printf("4 - Constant memory implementation \n");
    // printf("5 - Texture memory implementation \n ");
    int option;
    scanf("%d", &option);

    switch (option)
    {
    case 1:
        opencvConvolve("res/images/lena_bw.pgm");
        break;

    case 2:
        opencvCUDAConvolve();
        break;

        // case 3:
        //     imageConvolutionParallelSharedMemory(imageFilename, argv);
        //     break;

        // case 4:
        //     imageConvolutionParallelConstantMemory(imageFilename, argv);
        //     break;

        // case 5:
        //     imageConvolutionParallelTextureMomory(imageFilename, argv);
        //     break;

    default:
        printf("Incorrect input \n");
    }

    return 0;
}