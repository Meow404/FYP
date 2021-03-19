#ifndef IMAGECONVOLUTIONPARALLELSHAREDMEMORY
#define IMAGECONVOLUTIONPARALLELSHAREDMEMORY
#define KERNEL_OFFSET 3
// #define BLOCK_WIDTH 13

float *applyKernelToImageParallelSharedMemoryNoOverlap(float *image, int imageWidth, int imageHeight, kernel kernel, char *imagePath, int blockWidth);
// float applyKernelPerPixelSharedMemory(int y, int x, int kernelX, int kernelY, int imageWidth, int imageHeight, float *kernel, float *image);
__global__ void applyKernelPerPixelParallelSharedMemoryNoOverlap(int *kernelX, int *kernelY, int *imageWidth, int *imageHeight, float *kernel, float *image, float *sumArray);

float *applyKernelToImageParallelSharedMemoryNoOverlap(float *image, int imageWidth, int imageHeight, kernel kernel, char *imagePath, int blockWidth)
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

  int offsetX = (kernel.dimension - 1) / 2;
  int offsetY = (kernel.dimension - 1) / 2;

  int numHorBlocks = (imageWidth) / blockWidth;
  int numVerBlocks = (imageHeight) / blockWidth;

  if (imageWidth % blockWidth)
    numHorBlocks++;
  if (imageHeight % blockWidth)
    numVerBlocks++;

  dim3 dimGrid(numVerBlocks, numHorBlocks, 1);
  dim3 dimBlock(blockWidth, blockWidth, 1);
  applyKernelPerPixelParallelSharedMemoryNoOverlap<<<dimGrid, dimBlock, (blockWidth + offsetX) * (blockWidth + offsetY) * sizeFloat>>>(d_kernelDimensionX, d_kernelDimensionY, d_imageWidth, d_imageHeight, d_kernel, d_image, d_sumArray);

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

__global__ void applyKernelPerPixelParallelSharedMemoryNoOverlap(int *d_kernelDimensionX, int *d_kernelDimensionY, int *d_imageWidth, int *d_imageHeight, float *d_kernel, float *d_image, float *d_sumArray)
{
  // int comp = 45;
  int offsetX = (*d_kernelDimensionX - 1) / 2;
  int offsetY = (*d_kernelDimensionY - 1) / 2;

  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  // int y = blockIdx.y * blockDim.y + threadIdx.y;
  // int x = blockIdx.x * blockDim.x + threadIdx.x;

  int row = threadIdx.y;
  int col = threadIdx.x;

  extern __shared__ float local_imageSection[];
  // int imageIndex = y * (*d_imageWidth) + x;
  // local_imageSection[row][col] = d_image[y * (*d_imageWidth) + x - 2 * blockIdx.x];
  if ((row == 0 || row == blockDim.y - 1) && (col == 0 || col == blockDim.x - 1))
  {
    for (int i = y - offsetY; i <= y + offsetY; i++)
    {
      if (i < 0 || i >= *d_imageHeight)
        continue;
      for (int j = x - offsetX; j <= x + offsetX; j++)
      {
        if (j < 0 || j >= *d_imageHeight)
          continue;
        local_imageSection[(offsetY + row + i) * (blockDim.x) + (offsetX + col + j)] = d_image[i * (*d_imageWidth) + j];
      }
    }
  }
  else if ((row > offsetY || row < blockDim.y - offsetY) && (col == 0 || col == blockDim.x - 1))
  {

    for (int j = x - offsetX; j <= x + offsetX; j++)
    {
      if (j < 0 || j >= *d_imageHeight)
        continue;
      local_imageSection[(offsetY + row) * (blockDim.x) + (offsetX + col + j)] = d_image[y * (*d_imageWidth) + j];
    }
  }
  else if ((col > offsetX || col < blockDim.x - offsetX) && (row == 0 || row == blockDim.y - 1))
  {

    for (int i = y - offsetY; i <= y + offsetY; i++)
    {
      if (i < 0 || i >= *d_imageHeight)
        continue;
      local_imageSection[(offsetY + row + i) * (blockDim.x) + (offsetX + col)] = d_image[i * (*d_imageWidth) + x];
    }
  }
  else if ((col > offsetX || col < blockDim.x - offsetX) && (row > offsetY || row < blockDim.y - offsetY))
  {

    local_imageSection[(offsetY + row) * (blockDim.x) + (offsetX + col)] = d_image[y * (*d_imageWidth) + x];
  }

  __syncthreads();

  // if (blockIdx.x == 0 && blockidx.y ==0){
  //   printf("")
  // }

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
        float imageElement = local_imageSection[(row + j) * (blockDim.x) + col + i];

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

#endif
