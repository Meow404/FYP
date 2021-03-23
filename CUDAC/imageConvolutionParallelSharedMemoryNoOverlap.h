#ifndef IMAGECONVOLUTIONPARALLELSHAREDMEMORYNOOVERLAP
#define IMAGECONVOLUTIONPARALLELSHAREDMEMORYNOOVERLAP
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
  applyKernelPerPixelParallelSharedMemoryNoOverlap<<<dimGrid, dimBlock, (blockWidth + kernel.dimension - 1) * (blockWidth + kernel.dimension - 1) * sizeFloat>>>(d_kernelDimensionX, d_kernelDimensionY, d_imageWidth, d_imageHeight, d_kernel, d_image, d_sumArray);

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

  int memDimX = blockDim.x + *d_kernelDimensionX - 1;
  int memDimY = blockDim.y + *d_kernelDimensionY - 1;

  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  // int y = blockIdx.y * blockDim.y + threadIdx.y;
  // int x = blockIdx.x * blockDim.x + threadIdx.x;

  int row = threadIdx.y;
  int col = threadIdx.x;

  extern __shared__ float local_imageSection[];
  // int imageIndex = y * (*d_imageWidth) + x;
  // local_imageSection[row][col] = d_image[y * (*d_imageWidth) + x - 2 * blockIdx.x];
  // local_imageSection[row*(blockDim.x) + col] = d_image[y * (*d_imageWidth) + x];

  // if ((x - offsetX) < 0 || (y - offsetY) < 0)
  //   local_imageSection[row * (memDimX) + col] = 0;
  // else
  //   local_imageSection[row * (memDimX) + col] = d_image[(y - offsetY) * (*d_imageWidth) + (x - offsetX)];

  // if (row + blockDim.y < memDimY && col + blockDim.x < memDimX)
  //   if (x - offsetX + blockDim.x >= *d_imageWidth || y - offsetY + blockDim.y >= *d_imageHeight)
  //     local_imageSection[(row + blockDim.y) * (memDimX) + (col + blockDim.x)] = 0;
  //   else
  //     local_imageSection[(row + blockDim.y) * (memDimX) + (col + blockDim.x)] = d_image[(y - offsetY + blockDim.y) * (*d_imageWidth) + (x - offsetX + blockDim.x)];

  // if (row + blockDim.y < memDimY)
  //   if (x - offsetX < 0 || y - offsetY + blockDim.y >= *d_imageHeight)
  //     local_imageSection[(row + blockDim.y) * (memDimX) + col] = 0;
  //   else
  //     local_imageSection[(row + blockDim.y) * (memDimX) + col] = d_image[(y - offsetY + blockDim.y) * (*d_imageWidth) + (x - offsetX)];

  // if (col + blockDim.x < memDimX)
  //   if (x - offsetX + blockDim.x >= *d_imageWidth || y - offsetY < 0)
  //     local_imageSection[row * (memDimX) + (col + blockDim.x)] = 0;
  //   else
  //     local_imageSection[row * (memDimX) + (col + blockDim.x)] = d_image[(y - offsetY) * (*d_imageWidth) + (x - offsetX + blockDim.x)];

  __syncthreads();

  // convolution
  float sum = 0.0;
  for (int j = 0; j < *d_kernelDimensionY; j++)
  {
    for (int i = 0; i < *d_kernelDimensionX; i++)
    {
      float value;

      float k = d_kernel[i + j * (*d_kernelDimensionY)];
      float imageElement = local_imageSection[(row + j) * (memDimX) + col + i];

      value = k * imageElement;
      sum = sum + value;
    }
  }
  __syncthreads();
  d_sumArray[y * (*d_imageWidth) + x] = sum;
}
#endif
