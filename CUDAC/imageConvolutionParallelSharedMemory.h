#ifndef IMAGECONVOLUTIONPARALLELSHAREDMEMORY
#define IMAGECONVOLUTIONPARALLELSHAREDMEMORY
// #define KERNELDIMENSION 3
// #define BLOCK_WIDTH 13


float *applyKernelToImageParallelSharedMemory(float *image, int imageWidth, int imageHeight, kernel kernel, char *imagePath, int blockWidth);
// float applyKernelPerPixelSharedMemory(int y, int x, int kernelX, int kernelY, int imageWidth, int imageHeight, float *kernel, float *image);
__global__ void applyKernelPerPixelParallelSharedMemory(int *kernelX, int *kernelY, int *imageWidth, int *imageHeight, float *kernel, float *image, float *sumArray);
// void imageConvolutionParallelSharedMemory(const char *imageFilename, char **argv)
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
//     //Flip kernels to match convolution property and apply kernels to image
//     for (int i = 0; i < numKernels; i++)
//     {
//       applyKernelToImageParallelSharedMemory(hData, width, height, kernels[i], KERNELDIMENSION, imagePath);
//     }
//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);
//     float milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, start, stop);
//     printf("Time Shared Memory Parallel Implementation: %f \n", milliseconds);
//     totalTime += milliseconds;
//   }
//   printf("Time Serial Average Implementation: %f ms\n", totalTime / 10);
// }

float* applyKernelToImageParallelSharedMemory(float *image, int imageWidth, int imageHeight, kernel kernel, char *imagePath, int blockWidth)
{
  //printImage(image, imageWidth, imageHeight, "orginalImagePartition.txt");
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

  cudaMemcpy(d_kernelDimensionX, &kernel.dimension, sizeInt, cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernelDimensionY, &kernel.dimension, sizeInt, cudaMemcpyHostToDevice);
  cudaMemcpy(d_imageWidth, &imageWidth, sizeInt, cudaMemcpyHostToDevice);
  cudaMemcpy(d_imageHeight, &imageHeight, sizeInt, cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, kernel.matrix, kernel.dimension * kernel.dimension * sizeFloat, cudaMemcpyHostToDevice);
  cudaMemcpy(d_image, image, sizeImageArray, cudaMemcpyHostToDevice);

  int width = imageWidth * imageHeight;
  int numBlocks = (imageWidth) / blockWidth;

  // printf("image width %d image height %d \n ", imageWidth, imageHeight);
  // printf("kernel dimension %d \n", kernelDimension);

  int overlapX = (kernel.dimension + 1) / 2;
  int overlapY = (kernel.dimension + 1) / 2;

  int numHorBlocks = (imageWidth) / (blockWidth - overlapX);
  int numVerBlocks = (imageHeight) / (blockWidth - overlapY);

  if (imageWidth % (blockWidth - overlapX))
    numHorBlocks++;
  if (imageHeight % (blockWidth - overlapY))
    numVerBlocks++;

  // printf("Horizontal blocks %d vertical blocks %d \n\n", numHorBlocks, numVerBlocks);

  // int numHorBlocks = (imageWidth) / BLOCK_WIDTH;
  // int numVerBlocks = (imageHeight) / BLOCK_WIDTH;

  // if (imageWidth % BLOCK_WIDTH)
  //   numHorBlocks++;
  // if (imageHeight % BLOCK_WIDTH)
  //   numVerBlocks++;

  dim3 dimGrid(numVerBlocks, numHorBlocks, 1);
  dim3 dimBlock(blockWidth, blockWidth, 1);
  applyKernelPerPixelParallelSharedMemory<<<dimGrid, dimBlock>>>(d_kernelDimensionX, d_kernelDimensionY, d_imageWidth, d_imageHeight, d_kernel, d_image, d_sumArray);
  cudaError_t errSync = cudaGetLastError();
  cudaError_t errAsync = cudaDeviceSynchronize();
  if (errSync != cudaSuccess)
    printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
  if (errAsync != cudaSuccess)
    printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
  cudaMemcpy(sumArray, d_sumArray, sizeImageArray, cudaMemcpyDeviceToHost);

  return sumArray;
  //printImage(sumArray, imageWidth, imageHeight, "newImage2.txt");
  // char outputFilename[1024];
  // strcpy(outputFilename, imagePath);
  // strcpy(outputFilename + strlen(imagePath) - 4, "_sharedMemory_parallel_out.pgm");
  // sdkSavePGM(outputFilename, sumArray, imageWidth, imageHeight);

  //  for (int i = 0; i < imageHeight; i++) {
  //    printf("Line %d : ", i);
  //     for (int j = 0; j < imageWidth; j++) {
  //         printf("%2.2f ", sumArray[i*imageWidth+j]);
  //       //   if((i*width+j) % 15 == 0){
  //       //      printf("\n");
  //       //  }
  //     }
  //     printf("\n");
  // }
}

__global__ void applyKernelPerPixelParallelSharedMemory(int *d_kernelDimensionX, int *d_kernelDimensionY, int *d_imageWidth, int *d_imageHeight, float *d_kernel, float *d_image, float *d_sumArray)
{
  int comp = 45;
  int offsetX = (*d_kernelDimensionX - 1) / 2;
  int offsetY = (*d_kernelDimensionY - 1) / 2;

  int overlapX = (*d_kernelDimensionX + 1) / 2;
  int overlapY = (*d_kernelDimensionX + 1) / 2;

  int y = blockIdx.y * (blockDim.y - overlapX + 1) + threadIdx.y;
  int x = blockIdx.x * (blockDim.x - overlapY + 1) + threadIdx.x;
  // int y = blockIdx.y * blockDim.y + threadIdx.y;
  // int x = blockIdx.x * blockDim.x + threadIdx.x;

  int row = threadIdx.y;
  int col = threadIdx.x;

  const int blockWidthX = blockDim.x;
  const int blockWidthY = blockDim.Y;

  __shared__ float local_imageSection[blockWidthY][blockWidthX];
  int imageIndex = y * (*d_imageWidth) + x;
  // local_imageSection[row][col] = d_image[y * (*d_imageWidth) + x - 2 * blockIdx.x];
  local_imageSection[row][col] = d_image[y * (*d_imageWidth) + x];

  __syncthreads();

  if ((threadIdx.x >= offsetX || threadIdx.x < blockDim.x - offsetX + 1) && (threadIdx.y > offsetY || threadIdx.y < blockDim.y - offsetY + 1))
  {

    // if ((blockIdx.x == 0 || blockIdx.x == 0) && (blockIdx.y == 1 || blockIdx.y == 1))
    // printf("Block Id %d %d Thread Id %d %d \n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);

    if ((y < (*d_imageWidth)) && (x < (*d_imageWidth)))
    {
      float sum = 0.0;
      for (int j = 0; j < *d_kernelDimensionY; j++)
      {
        //Ignore out of bounds
        if (row + j - offsetY < 0)
        {
          continue;
        }

        for (int i = 0; i < *d_kernelDimensionX; i++)
        {
          //Ignore out of bounds
          if (col + i - offsetX < 0)
          {
            continue;
          }

          float value;

          float k = d_kernel[i + j * (*d_kernelDimensionY)];
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
      d_sumArray[y * (*d_imageWidth) + x] = sum;
    }
  }
}

#endif
