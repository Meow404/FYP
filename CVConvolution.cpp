#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
 // Read the image file
 cv::Mat image = imread("res\\images\\Lenna.png");
 cv::Mat result;
 const int ratio = 3;
 const int lowThreshold = 20;
 const int kernel_size = 3;

 // Check for failure
 if (image.empty()) 
 {
  cout << "Could not open or find the image" << endl;
  cin.get(); //wait for any key press
  return -1;
 }
 
 cv::cvtColor(image, result, cv::COLOR_BGR2GRAY);
 cv::blur(result, result, cv::Size(3, 3));
 cv::Canny(result, result, lowThreshold, lowThreshold*ratio, kernel_size);
//  image.copyTo(result);
 cout << "image = " << endl << " "  << image << endl << endl;
 cout << "result = " << endl << " "  << result << endl << endl;

 return 0;
}