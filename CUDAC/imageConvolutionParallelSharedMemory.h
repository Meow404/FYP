#ifndef IMAGECONVOLUTIONPARALLELSHAREDMEMORY
#define IMAGECONVOLUTIONPARALLELSHAREDMEMORY
#define KERNEL_OFFSET 3
// #define BLOCK_WIDTH 13

float *applyKernelToImageParallelSharedMemory(float *image, int imageWidth, int imageHeight, kernel kernel, char *imagePath, int blockWidth);
// float applyKernelPerPixelSharedMemory(int y, int x, int kernelX, int kernelY, int imageWidth, int imageHeight, float *kernel, float *image);
__global__ void applyKernelPerPixelParallelSharedMemory(int *kernelX, int *kernelY, int *imageWidth, int *imageHeight, float *kernel, float *image, float *sumArray);

float *applyKernelToImageParallelSharedMemory(float *image, int imageWidth, int imageHeight, kernel kernel, char *imagePath, int blockWidth)
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

  int overlapX = (kernel.dimension - 1);
  int overlapY = (kernel.dimension - 1);

  int numHorBlocks = (imageWidth) / (blockWidth - overlapX);
  int numVerBlocks = (imageHeight) / (blockWidth - overlapY );

  if (imageWidth % (blockWidth - overlapX))
    numHorBlocks++;
  if (imageHeight % (blockWidth - overlapY))
    numVerBlocks++;

  dim3 dimGrid(numVerBlocks, numHorBlocks, 1);
  dim3 dimBlock(blockWidth, blockWidth, 1);
  applyKernelPerPixelParallelSharedMemory<<<dimGrid, dimBlock, blockWidth*blockWidth*sizeFloat>>>(d_kernelDimensionX, d_kernelDimensionY, d_imageWidth, d_imageHeight, d_kernel, d_image, d_sumArray);

  cudaError_t errSync = cudaGetLastError();
  cudaError_t errAsync = cudaDeviceSynchronize();
  if (errSync != cudaSuccess)
    printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
  if (errAsync != cudaSuccess)
    printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));

  cudaMemcpy(sumArray, d_sumArray, sizeImageArray, cudaMemcpyDeviceToHost);

  // CUDA free varibles
  cudaFree(d_kernelDimensionX);
  cudaFree(d_kernelDimensionY);
  cudaFree(d_imageWidth);
  cudaFree(d_imageHeight);
  cudaFree(d_kernel);
  cudaFree(d_image);
  cudaFree(d_sumArray);

  return sumArray;
}

__global__ void applyKernelPerPixelParallelSharedMemory(int *d_kernelDimensionX, int *d_kernelDimensionY, int *d_imageWidth, int *d_imageHeight, float *d_kernel, float *d_image, float *d_sumArray)
{
  // int comp = 45;
  int offsetX = (*d_kernelDimensionX - 1) / 2;
  int offsetY = (*d_kernelDimensionY - 1) / 2;

  int overlapX = (*d_kernelDimensionX - 1);
  int overlapY = (*d_kernelDimensionY - 1);

  int y = blockIdx.y * (blockDim.y - overlapX) + threadIdx.y;
  int x = blockIdx.x * (blockDim.x - overlapY) + threadIdx.x;
  // int y = blockIdx.y * blockDim.y + threadIdx.y;
  // int x = blockIdx.x * blockDim.x + threadIdx.x;

  int row = threadIdx.y;
  int col = threadIdx.x;

  extern __shared__ float local_imageSection[];
  // int imageIndex = y * (*d_imageWidth) + x;
  // local_imageSection[row][col] = d_image[y * (*d_imageWidth) + x - 2 * blockIdx.x];
  local_imageSection[row*(blockDim.x) + col] = d_image[y * (*d_imageWidth) + x];

  __syncthreads();

  if ((threadIdx.x > offsetX || threadIdx.x < blockDim.x - offsetX) && (threadIdx.y > offsetY || threadIdx.y < blockDim.y - offsetY))
  {

    // if ((blockIdx.x == 0 || blockIdx.x == 0) && (blockIdx.y == 1 || blockIdx.y == 1))
    // printf("Block Id %d %d Thread Id %d %d \n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);

    if ((y < (*d_imageHeight)) && (x < (*d_imageWidth)))
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
          float imageElement = local_imageSection[(row + j - offsetY)*(blockDim.x) + col + i - offsetX];

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
