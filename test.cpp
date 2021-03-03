#include "kernelHandler.h"

int main()
{
    kernelHandler K = kernelHandler("kernels.txt");

    K.printKernel();
}