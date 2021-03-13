#ifndef IMAGECONVOLUTIONPARALLELSHAREDCONSTANTMEMORY
#define IMAGECONVOLUTIONPARALLELSHAREDCONSTANTMEMORY
// #define KERNEL_DIMENSION 3
// #define BLOCK_WIDTH 13


float *applyKernelToImageParallelSharedConstantMemory(float *image, int imageWidth, int imageHeight, kernel kernel, char *imagePath, int blockWidth);
// float applyKernelPerPixelSharedConstantMemory(int y, int x, int kernelX, int kernelY, int imageWidth, int imageHeight, float *kernel, float *image);
__global__ void applyKernelPerPixelParallelSharedConstantMemory(float *d_image, float *d_sumArray);

// __constant__ float kernelConstant[128 * 129];
// __constant__ int imageWidthConstant;
// __constant__ int imageHeightConstant;
// __constant__ int kernelDimensionXConstant;
// __constant__ int kernelDimensionYConstant;

// void imageConvolutionParallelSharedConstantMemory(const char *imageFilename, char **argv)
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
//   int kernelDimension = 3;

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
//       applyKernelToImageParallelSharedConstantMemory(hData, width, height, kernels[i], kernelDimension, imagePath);
//     }
//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);
//     float milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, start, stop);
//     printf("Time Constant Implementation: %f \n", milliseconds);
//     totalTime += milliseconds;
//   }
//   printf("Time Serial Average Implementation: %f ms\n", totalTime / 10);
// }

float *applyKernelToImageParallelSharedConstantMemory(float *image, int imageWidth, int imageHeight, kernel kernel, char *imagePath, int blockWidth)
{
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
  cudaMalloc((void **)&d_kernel, kernel.dimension * kernel.dimension * sizeFloat);
  cudaMalloc((void **)&d_image, sizeImageArray);
  cudaMalloc((void **)&d_sumArray, sizeImageArray);

  cudaMemcpy(d_image, image, sizeImageArray, cudaMemcpyHostToDevice);

  //constants
  cudaMemcpyToSymbol(kernelConstant, kernel.matrix, sizeof(float) * kernel.dimension * kernel.dimension);
  cudaMemcpyToSymbol(imageWidthConstant, &imageWidth, sizeInt);
  cudaMemcpyToSymbol(imageHeightConstant, &imageHeight, sizeInt);
  cudaMemcpyToSymbol(kernelDimensionXConstant, &kernel.dimension, sizeInt);
  cudaMemcpyToSymbol(kernelDimensionYConstant, &kernel.dimension, sizeInt);

  int overlapX = (kernel.dimension + 1) / 2;
  int overlapY = (kernel.dimension + 1) / 2;

  int numHorBlocks = (imageWidth) / (blockWidth - overlapX);
  int numVerBlocks = (imageHeight) / (blockWidth - overlapY);

  if (imageWidth % (blockWidth - overlapX))
    numHorBlocks++;
  if (imageHeight % (blockWidth - overlapY))
    numVerBlocks++;

  dim3 dimGrid(numVerBlocks, numHorBlocks, 1);
  dim3 dimBlock(blockWidth, blockWidth, 1);
  applyKernelPerPixelParallelSharedConstantMemory<<<dimGrid, dimBlock>>>(d_image, d_sumArray);
  cudaMemcpy(sumArray, d_sumArray, sizeImageArray, cudaMemcpyDeviceToHost);

  return sumArray;
  // char outputFilename[1024];
  // strcpy(outputFilename, imagePath);
  // strcpy(outputFilename + strlen(imagePath) - 4, "_shared_constant_memory_parallel_out.pgm");
  // sdkSavePGM(outputFilename, sumArray, imageWidth, imageHeight);
}
__global__ void applyKernelPerPixelParallelSharedConstantMemory(float *d_image, float *d_sumArray)
{
  int comp = 45;
  int offsetX = (kernelDimensionXConstant - 1) / 2;
  int offsetY = (kernelDimensionYConstant - 1) / 2;

  int overlapX = (kernelDimensionXConstant + 1) / 2;
  int overlapY = (kernelDimensionYConstant + 1) / 2;

  int y = blockIdx.y * (blockDim.y - offsetX + 1) + threadIdx.y;
  int x = blockIdx.x * (blockDim.x - offsetY + 1) + threadIdx.x;
  // int y = blockIdx.y * blockDim.y + threadIdx.y;
  // int x = blockIdx.x * blockDim.x + threadIdx.x;

  int row = threadIdx.y;
  int col = threadIdx.x;

  __shared__ float local_imageSection[28][28];
  int imageIndex = y * (imageWidthConstant) + x;
  // local_imageSection[row][col] = d_image[y * (*d_imageWidth) + x - 2 * blockIdx.x];
  local_imageSection[row][col] = d_image[y * (imageWidthConstant) + x];

  __syncthreads();

  //Need to fill in if statement ******
  if ((threadIdx.x >= offsetX || threadIdx.x < blockDim.x - offsetX + 1) && (threadIdx.y > offsetY || threadIdx.y < blockDim.y - offsetY + 1))
  {

    // if ((blockIdx.x == 0 || blockIdx.x == 0) && (blockIdx.y == 1 || blockIdx.y == 1))
    // printf("Block Id %d %d Thread Id %d %d \n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);

    if ((y < (imageHeightConstant)) && (x < (imageWidthConstant)))
    {
      float sum = 0.0;
      for (int j = 0; j < kernelDimensionYConstant; j++)
      {
        //Ignore out of bounds
        if (row + j - offsetY < 0)
        {
          continue;
        }

        for (int i = 0; i < kernelDimensionXConstant; i++)
        {
          //Ignore out of bounds
          if (
              col + i - offsetX < 0)
          {
            continue;
          }
          float value;

          float k = kernelConstant[i + j * (kernelDimensionYConstant)];
          float imageElement = local_imageSection[row + j - offsetY][col + i - offsetX];

          value = k * imageElement;
          sum = sum + value;
        }
      }

      //Normalising output ;
      // if (sum < 0)
      //   sum = 0;
      // else if (sum > 1)
      //   sum = 1;
      __syncthreads();
      // d_sumArray[y * (*d_imageWidth) + x - 2 * blockIdx.x] = sum;
      d_sumArray[y * (imageWidthConstant) + x] = sum;
    }
  }
}

#endif
