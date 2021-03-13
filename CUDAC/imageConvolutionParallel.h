#ifndef IMAGECONVOLUTIONPARALLEL
#define IMAGECONVOLUTIONPARALLEL


float* applyKernelToImageParallelNaive(float *image, int imageWidth, int imageHeight, kernel* kernel, char *imagePath, int blockWidth);
__global__ void applyKernelPerPixelParallel(int *kernelX, int *kernelY, int *imageWidth, int *imageHeight, float *kernel, float *image, float *sumArray);

// void imageConvolutionParallel(const char *imageFilename, char **argv)
// {
//   // load image from disk
//   float *hData = NULL;
//   unsigned int width, height;
//   char *imagePath = sdkFindFilePath(imageFilename, argv[0]);

//   if (imagePath == NULL)
//   {
//     printf("Unable to source image file: %s\n", imageFilename);
//     exit(EXIT_FAILURE);
//   }

//   sdkLoadPGM(imagePath, &hData, &width, &height);
//   printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);

//   //Get Kernels
//   FILE *fp = fopen("kernels.txt", "r");
//   if (fp == NULL)
//   {
//     perror("Error in opening file");
//     exit(EXIT_FAILURE);
//   }
//   int numKernels = getNumKernels(fp);
//   // int kernelDimension = 3;

//   float **kernels = (float **)malloc(sizeof(float *) * numKernels);
//   for (int i = 0; i < numKernels; i++)
//   {
//     kernels[i] = (float *)malloc(sizeof(float) * 100);
//   }
//   loadAllKernels(kernels, fp);
//   fclose(fp);
//   float totalTime = 0.0;
//   for (int i = 0; i < 10; i++)
//   {
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start);

//     for (int i = 0; i < numKernels; i++)
//     {
//       applyKernelToImageParallelNaive(hData, width, height, kernels[i], KERNELDIMENSION, imagePath);
//     }
//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);
//     float milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, start, stop);
//     printf("Time Naive Parallel Implementation: %f \n", milliseconds);
//     totalTime += milliseconds;
//   }
//   printf("Time Serial Average Implementation: %f ms\n", totalTime/10);
// }

float* applyKernelToImageParallelNaive(float *image, int imageWidth, int imageHeight, kernel* kernel, char *imagePath, int blockWidth)
{
        for (int i = 0; i < kernel->dimension; i++)
        {
            printf("\n");
            for (int j = 0; j < kernel->dimension; j++)
            {
                printf("%f ", kernel->matrix[i*kernel->dimension + j]);
            }
        }
  
  int *d_kernelDimensionX, *d_kernelDimensionY, *d_imageWidth, *d_imageHeight;
  float *d_kernel, *d_image, *d_sumArray;

  int sizeInt = sizeof(int);
  int sizeFloat = sizeof(float);
  int sizeImageArray = imageWidth * imageHeight * sizeFloat;
  float *sumArray = (float *)malloc(sizeImageArray);

  cudaMalloc((void **)&d_kernelDimensionX, sizeInt);
  cudaMalloc((void **)&d_kernelDimensionY, sizeInt);
  cudaMalloc((void **)&d_imageWidth, sizeInt);
  cudaMalloc((void **)&d_imageHeight, sizeInt);
  cudaMalloc((void **)&d_kernel, kernel->dimension * kernel->dimension * sizeFloat);
  cudaMalloc((void **)&d_image, sizeImageArray);
  cudaMalloc((void **)&d_sumArray, sizeImageArray);

  cudaMemcpy(d_kernelDimensionX, &kernel->dimension, sizeInt, cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernelDimensionY, &kernel->dimension, sizeInt, cudaMemcpyHostToDevice);
  cudaMemcpy(d_imageWidth, &imageWidth, sizeInt, cudaMemcpyHostToDevice);
  cudaMemcpy(d_imageHeight, &imageHeight, sizeInt, cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, kernel->matrix, kernel->dimension * kernel->dimension * sizeFloat, cudaMemcpyHostToDevice);
  cudaMemcpy(d_image, image, sizeImageArray, cudaMemcpyHostToDevice);

  int numHorBlocks = imageWidth / blockWidth;
  int numVerBlocks = imageHeight / blockWidth;

  if (imageWidth % blockWidth)
    numHorBlocks++;
  if (imageHeight % blockWidth)
    numVerBlocks++;

  dim3 dimGrid(numVerBlocks, numHorBlocks, 1);
  dim3 dimBlock(blockWidth, blockWidth, 1);

  applyKernelPerPixelParallel<<<dimGrid, dimBlock>>>(d_kernelDimensionX, d_kernelDimensionY, d_imageWidth, d_imageHeight, d_kernel, d_image, d_sumArray);
  cudaMemcpy(sumArray, d_sumArray, sizeImageArray, cudaMemcpyDeviceToHost);

  //printImage(sumArray,imageWidth,imageHeight,"newImageP.txt");

  return sumArray;
  // char outputFilename[1024];
  // strcpy(outputFilename, imagePath);
  // strcpy(outputFilename + strlen(imagePath) - 4, "_parallel_out.pgm");
  // sdkSavePGM(outputFilename, sumArray, imageWidth, imageHeight);
}
__global__ void applyKernelPerPixelParallel(int *d_kernelDimensionX, int *d_kernelDimensionY, int *d_imageWidth, int *d_imageHeight, float *d_kernel, float *d_image, float *d_sumArray)
{

  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  int offsetX = (*d_kernelDimensionX - 1) / 2;
  int offsetY = (*d_kernelDimensionY - 1) / 2;

  float sum = 0.0;

  if ((y < (*d_imageWidth)) && (x < (*d_imageWidth)))
  {
    for (int j = 0; j < *d_kernelDimensionY; j++)
    {
      //Ignore out of bounds
      if (y + j < offsetY || y + j - offsetY >= *d_imageHeight)
        continue;

      for (int i = 0; i < *d_kernelDimensionX; i++)
      {
        //Ignore out of bounds
        if (x + i < offsetX || x + i - offsetX >= *d_imageWidth)
          continue;

        float k = d_kernel[i + j * (*d_kernelDimensionY)];
        float imageElement = d_image[y * (*d_imageWidth) + x + i - offsetX + (*d_imageWidth) * (j - 1)];
        float value = k * imageElement;
        sum = sum + value;
      }
    }
    int imageIndex = y * (*d_imageWidth) + x;
    // //Normalising output
    // if (sum < 0)
    //   sum = 0;
    // else if (sum > 1)
    //   sum = 1;
    d_sumArray[y * (*d_imageWidth) + x] = sum;
  }
}
#endif
