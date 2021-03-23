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
#include "imageConvolution2DTextureMemory.h"
#include "imageConvolutionParallelSharedMemoryNoOverlap.h"
#include "imageConvolutionParallelSharedConstantMemoryNoOverlap.h"

const char *imageFilename = "res//images//1024_lena_bw.pgm";
//const char *imageFilename = "galaxy.ascii.pgm";
#define ITERATIONS 100
#define BLOCK_WIDTH 24
#define FILE_INDEX 3

float *imageConvolutionParallel(const char *imageFilename, char **argv, int option, bool print_save = true)
{
  // load image from disk
  float *hData = NULL;
  char buf[512];
  unsigned int width, height;
  char *imagePath = sdkFindFilePath(imageFilename, argv[0]);
  float *results;

  if (imagePath == NULL)
  {
    printf("Unable to source image file: %s\n", imageFilename);
    exit(EXIT_FAILURE);
  }

  sdkLoadPGM(imagePath, &hData, &width, &height);
  if (print_save)
    printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);

  FILE *fp = fopen("kernels.txt", "r");
  if (fp == NULL)
  {
    perror("Error in opening file");
    exit(EXIT_FAILURE);
  }

  int numOfKernels;
  fgets(buf, sizeof(buf), fp);
  sscanf(buf, "%d", &numOfKernels);

  // printf("%d kernel to be loaded\n", numOfKernels);

  kernel **kernels = loadAllKernels(fp, numOfKernels);
  // printKernels(kernels, numOfKernels);

  // kernelHandler kh = kernelHandler("../kernels.txt");

  //printf("Kernels loaded\n");

  results = (float *)malloc(numOfKernels * sizeof(float));

  //Get Kernels
  // FILE *fp = fopen("kernels.txt", "r");
  // if (fp == NULL)
  // {
  //   perror("Error in opening file");
  //   exit(EXIT_FAILURE);
  // }

  for (int i = 0; i < numOfKernels; i++)
  {

    char outputFilename[1024];
    char *file_name;

    float *result;
    float totalTime = 0.0;
    if (print_save)
      printf("\n\n\nKernel Dimension : %dx%d\n", kernels[i]->dimension, kernels[i]->dimension);

    int j = 0;
    for (; j < ITERATIONS; j++)
    {
      cudaDeviceReset();
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start);

      switch (option)
      {
      case 1:
        result = applyKernelToImageSerial(hData, width, height, *kernels[i], imagePath);
        if (j == 0)
          sprintf(file_name, "_%dx%d_serial_out.pgm", kernels[i]->dimension, kernels[i]->dimension);
        break;

      case 2:
        result = applyKernelToImageParallelNaive(hData, width, height, kernels[i], imagePath, BLOCK_WIDTH);
        if (j == 0)
          sprintf(file_name, "_%dx%d_parallel_out.pgm", kernels[i]->dimension, kernels[i]->dimension);
        break;

      case 3:
        result = applyKernelToImageParallelSharedMemory(hData, width, height, *kernels[i], imagePath, BLOCK_WIDTH);
        if (j == 0)
          sprintf(file_name, "_%dx%d_parallel_shared_out.pgm", kernels[i]->dimension, kernels[i]->dimension);
        break;

      case 4:
        result = applyKernelToImageParallelConstantMemory(hData, width, height, *kernels[i], imagePath, BLOCK_WIDTH);
        if (j == 0)
          sprintf(file_name, "_%dx%d_parallel_constant_out.pgm", kernels[i]->dimension, kernels[i]->dimension);
        break;

      case 5:
        result = applyKernelToImageParallelSharedConstantMemory(hData, width, height, *kernels[i], imagePath, BLOCK_WIDTH);
        if (j == 0)
          sprintf(file_name, "_%dx%d_parallel_shared_constant_out.pgm", kernels[i]->dimension, kernels[i]->dimension);
        break;

      case 6:
        result = applyKernelToImageParallelTextureMomory(hData, width, height, *kernels[i], imagePath, BLOCK_WIDTH);
        if (j == 0)
          sprintf(file_name, "_%dx%d_parallel_texture_out.pgm", kernels[i]->dimension, kernels[i]->dimension);
        break;

      case 7:
        result = applyKernelToImageParallel2DTextureMomory(hData, width, height, *kernels[i], imagePath, BLOCK_WIDTH);
        if (j == 0)
          sprintf(file_name, "_%dx%d_paralled_2D_texture_out.pgm", kernels[i]->dimension, kernels[i]->dimension);
        break;
      case 8:
        result = applyKernelToImageParallelSharedMemory(hData, width, height, *kernels[i], imagePath, BLOCK_WIDTH + kernels[i]->dimension - 1);
        if (j == 0)
          sprintf(file_name, "_%dx%d_parallel_shared_mod_out.pgm", kernels[i]->dimension, kernels[i]->dimension);
        break;
      case 9:
        result = applyKernelToImageParallelSharedConstantMemory(hData, width, height, *kernels[i], imagePath, BLOCK_WIDTH + kernels[i]->dimension - 1);
        if (j == 0)
          sprintf(file_name, "_%dx%d_parallel_shared_constant_mod_out.pgm", kernels[i]->dimension, kernels[i]->dimension);
        break;
      case 10:
        result = applyKernelToImageParallelSharedMemoryNoOverlap(hData, width, height, *kernels[i], imagePath, BLOCK_WIDTH);
        if (j == 0)
          sprintf(file_name, "_%dx%d_parallel_shared_no_overlap_out.pgm", kernels[i]->dimension, kernels[i]->dimension);
        break;
      case 11:
        result = applyKernelToImageParallelSharedConstantMemoryNoOverlap(hData, width, height, *kernels[i], imagePath, BLOCK_WIDTH);
        if (j == 0)
          sprintf(file_name, "_%dx%d_parallel_shared_constant_no_overlap_out.pgm", kernels[i]->dimension, kernels[i]->dimension);
        break;

      default:
        printf("Incorrect input \n");
      }
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);
      if (print_save)
        printf("[%3d] Time : %f \n", j, milliseconds);
      totalTime += milliseconds;

      if (totalTime > 300000)
      {
        j++;
        printf("Completed %d iteration.... exiting\n", j);
        break;
      }

      if (j != ITERATIONS - 1)
        free(result);
    }
    if (print_save)
      printf("%d | %f \n", kernels[i]->dimension, totalTime / j);

    results[i] = totalTime / j;
    if (print_save)
    {
      for (int j = 0; j < height; j++)
      {
        printf("[%3d] : ", j);
        for (int i = 0; i < width; i++)
        {
          printf(" |%5.2f|", result[j * width + i]);
        }
        printf("\n");
      }
      strcpy(outputFilename, imagePath);
      strcpy(outputFilename + strlen(imagePath) - 4, file_name);
      printf("Saving to %s", outputFilename);
      sdkSavePGM(outputFilename, result, width, height);
      free(result);
    }
  }
  freeKernels(kernels, numOfKernels);
  free(hData);
  return results;
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
  printf("6 - Texture memory implementation \n");
  printf("7 - 2D Texture memory implementation \n");
  printf("8 - Shared memory modified implementation \n");
  printf("9 - Shared + constant memory modified implementation \n");
  printf("10 - Shared memory No Overlap implementation \n");
  printf("11 - Shared Constant memory No Overlap implementation \n");
  printf("12 - All \n ");
  int option;
  char buf[512];
  scanf("%d", &option);

  // kernelHandler kh = kernelHandler("../kernels.txt");

  //printf("Kernels loaded\n");

  char image_files[5][35] = {"res//images//256_lena_bw.pgm", "res//images//lena_bw.pgm", "res//images//1024_lena_bw.pgm", "res//images//2048_lena_bw.pgm"};

  if (option < 12)
    imageConvolutionParallel(image_files[FILE_INDEX], argv, option);
  else if (option == 12)
  {

    for (int k = 0; k < 4; k++)
    {

      FILE *fp = fopen("kernels.txt", "r");
      if (fp == NULL)
      {
        perror("Error in opening file");
        exit(EXIT_FAILURE);
      }

      int numOfKernels;
      fgets(buf, sizeof(buf), fp);
      sscanf(buf, "%d", &numOfKernels);
      kernel **kernels = loadAllKernels(fp, numOfKernels);

      float *results;
      for (int i = 1; i < 12; i++)
      {
        if (i > 9)
        {
          results = imageConvolutionParallel(image_files[k], argv, i, false);
          for (int j = 0; j < numOfKernels; j++)
          {
            printf("\n%8.3f", results[j]);
          }
          printf("\n");
        }
        printf("Image %d : Type %d DONE\n", k, i);
      }
      // printf("Image : %s\n", image_files[k]);
      // printf("| MxM | Serial |Parallel| Shared |Constant|   SC   |  Text  | 2DText |\n");
      // for (int i = 0; i < numOfKernels; i++)
      // {
      //   printf("|%2dx%2d|", kernels[i]->dimension, kernels[i]->dimension);
      //   for (int j = 1; j < 12; j++)
      //   {
      //     if (i != 8 && i != 9)
      //     printf("%8.3f|", results[j - 1][i]);
      //   }
      //   printf("\n");
      // }
      printf("=================================\n\n\n");
    }
  }
  else
    printf("\n\nInvalid Input !!!");
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
