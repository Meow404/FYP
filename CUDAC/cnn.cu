#include <iostream>
#include <stdio.h>
#include <helper_functions.h>
#include <helper_cuda.h>
// #include "../kernelHandler.h"
#include "helpers.h"
#include "imageConvolutionSerial.h"
#include "imageConvolutionParallel.h"
#include "imageConvolutionParallelSharedMemory.h"
#include "imageConvolutionParallelConstantMemory.h"
#include "imageConvolutionParallelSharedConstantMemory.h"
#include "imageConvolutionTextureMemory.h"

const char *imageFilename = "res//images//lena_bw.pgm";
//const char *imageFilename = "galaxy.ascii.pgm";
#define ITERATIONS 100
#define BLOCK_WIDTH 13

void imageConvolutionParallel(const char *imageFilename, char **argv, int option)
{
  // load image from disk
  float *hData = NULL;
  char buf[512];
  unsigned int width, height;
  char *imagePath = sdkFindFilePath(imageFilename, argv[0]);

  if (imagePath == NULL)
  {
    printf("Unable to source image file: %s\n", imageFilename);
    exit(EXIT_FAILURE);
  }

  sdkLoadPGM(imagePath, &hData, &width, &height);
  printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);

  //Get Kernels
  // FILE *fp = fopen("kernels.txt", "r");
  // if (fp == NULL)
  // {
  //   perror("Error in opening file");
  //   exit(EXIT_FAILURE);
  // }

  FILE *fp = fopen("kernels.txt", "r");
  if (fp == NULL)
  {
    perror("Error in opening file");
    exit(EXIT_FAILURE);
  }

  int numOfKernels;
  fgets(buf, sizeof(buf), fp);
  sscanf(buf, "%d", &numOfKernels);

  kernel** kernels = loadAllKernels(fp, numOfKernels);

  // kernelHandler kh = kernelHandler("../kernels.txt");

  for (int i = 0; i < numOfKernels; i++)
  {
    float totalTime = 0.0;
    printf("Kernel Dimension : %dx%d", kernels[i]->dimension, kernels[i]->dimension);

    for (int i = 0; i < ITERATIONS; i++)
    {
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start);

      switch (option)
      {
      case 1:
        applyKernelToImageSerial(hData, width, height, *kernels[i], imagePath);
        break;

      case 2:
        applyKernelToImageParallelNaive(hData, width, height, *kernels[i], imagePath, BLOCK_WIDTH);
        break;

      case 3:
        applyKernelToImageParallelSharedMemory(hData, width, height, *kernels[i], imagePath, BLOCK_WIDTH);
        break;

      case 4:
        applyKernelToImageParallelConstantMemory(hData, width, height, *kernels[i], imagePath, BLOCK_WIDTH);
        break;

      case 5:
        applyKernelToImageParallelSharedConstantMemory(hData, width, height, *kernels[i], imagePath, BLOCK_WIDTH);
        break;

      case 6:
        applyKernelToImageParallelTextureMomory(hData, width, height, *kernels[i], imagePath, BLOCK_WIDTH);
        break;

      default:
        printf("Incorrect input \n");
      }
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);
      printf("Time Naive Parallel Implementation: %f \n", milliseconds);
      totalTime += milliseconds;
    }
    printf("Time Serial Average Implementation: %f ms\n", totalTime / 10);
  }
}

int main(int argc, char **argv)
{
  printf("Image convolution project \n");
  printf("Please select an option \n");
  printf("1 - Serial Implementation \n");
  printf("2 - Naive parallel implementation \n");
  printf("3 - Shared memory implementation \n");
  printf("4 - Constant memory implementation \n");
  printf("5 - Shared Constant memory implementation \n");
  printf("6 - Texture memory implementation \n ");
  int option;
  scanf("%d", &option);

  imageConvolutionParallel(imageFilename, argv, option);
  // switch (option)
  // {
  // case 1:
  //   imageConvolutionSerial(imageFilename, argv);
  //   break;

  // case 2:
  //   imageConvolutionParallel(imageFilename, argv);
  //   break;

  // case 3:
  //   imageConvolutionParallelSharedMemory(imageFilename, argv);
  //   break;

  // case 4:
  //   imageConvolutionParallelConstantMemory(imageFilename, argv);
  //   break;

  // case 5:
  //   imageConvolutionParallelSharedConstantMemory(imageFilename, argv);
  //   break;

  // case 6:
  //   imageConvolutionParallelTextureMomory(imageFilename, argv);
  //   break;

  // default:
  //   printf("Incorrect input \n");
  // }

  return 0;
}
