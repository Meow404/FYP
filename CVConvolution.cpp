#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include "kernelHandler.h"

using namespace cv;
using namespace std;

int main(int argc, char **argv)
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
    for (kernel k : kh.getKernels())
    {
        cv::Mat ker = cv::Mat(k.dimension, k.dimension, CV_16F, &k.matrix);
        cout << "kernel = " << ker;
        cv::normalize(ker, ker, 1.0, 0.0, NORM_L1);	
        cout << "kernel = " << ker;

        cv::filter2D(result, result, -1, ker, Point(-1, -1), 5.0, BORDER_REPLICATE);
        cout << "result = " << endl
             << " " << result << endl
             << endl;
    }
    //  image.copyTo(result);

    return 0;
}