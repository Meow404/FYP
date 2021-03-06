#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

struct kernel
{
    int dimension;
    float **matrix;
};

class kernelHandler
{
    std::vector<kernel> kernels;
    int numOfKernels;
    const char *kernelFilename;

    void loadAllKernels();
    void loadRow(float *row, int kernelDimension, string buf);

public:
    kernelHandler(const char *kernelFilename);
    void printKernel(); 
    int getNumOfKernels();
    kernel getKernel(int index);
    std::vector<kernel> getKernels();
    cv::Mat returnMatrix(int index);
};