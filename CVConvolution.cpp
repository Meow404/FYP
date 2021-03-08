#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "opencv2/cudaarithm.hpp"
#include "opencv2/core/cuda.hpp"
#include <iostream>
#include "kernelHandler.h"

using namespace cv;
using namespace std;

int opencvConvolve()
{
    // Read the image file
    cv::Mat image = imread("res/images/lena_bw.pgm");
    cv::Mat result;
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

    cv::cvtColor(image, result, cv::COLOR_BGR2GRAY);
    kh.printKernel();
    for (int i = 0; i < kh.getNumOfKernels(); i++)
    {
        int dim = kh.getKernel(i).dimension;
        cv::Mat k = kh.returnMatrix(i);
        cout << "kernel = " << k << endl;
        cv::normalize(k, k, 1.0, 0.0, NORM_L1);
        cout << "kernel = " << k << endl;

        // Ptr<cuda::Convolution> convolver = cuda::createConvolution(k.size);
        // convolver->convolve(result, k, result);
        cv::filter2D(result, result, -1, k, Point(-1, -1), 5.0, BORDER_REPLICATE);
        cout << "result = " << endl
             << " " << result << endl
             << endl;
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
    
    gpu_image.upload(image);

    cv::cuda::cvtColor(gpu_image, gpu_result, cv::COLOR_BGR2GRAY);
    kh.printKernel();
    for (int i = 0; i < kh.getNumOfKernels(); i++)
    {
        int dim = kh.getKernel(i).dimension;
        cv::Mat k = kh.returnMatrix(i);
        cout << "kernel = " << k << endl;
        gpu_kernel.upload(k);
        // cv::normalize(ker, ker, 1.0, 0.0, NORM_L1);
        // cout << "kernel = " << ker;

        Ptr<cuda::Convolution> convolver = cuda::createConvolution(cv::Size(dim, dim));
        convolver->convolve(gpu_result, gpu_kernel, gpu_result);
        // cv::filter2D(result, result, -1, kernel, Point(-1, -1), 5.0, BORDER_REPLICATE);
        gpu_result.download(result);
        cout << "result = " << endl
             << " " << result << endl
             << endl;
    }
    //  image.copyTo(result);
}

int main(int argc, char **argv)
{
    printf("Image convolution project \n");
    printf("Please select an option \n");
    printf("1 - Serial OpenCV Implementation \n");
    printf("2 - OpenCV CUDA implementation \n");
    // printf("3 - Shared memory implementation \n");
    // printf("4 - Constant memory implementation \n");
    // printf("5 - Texture memory implementation \n ");
    int option;
    scanf("%d", &option);

    switch (option)
    {
    case 1:
        opencvConvolve();
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