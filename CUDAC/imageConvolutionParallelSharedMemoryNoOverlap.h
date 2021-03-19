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
  xt = x - offsetX;
  yt = y - offsetY;
// ....if ( xt < 0 || yt < 0 )
// ........data[threadIdx.x][threadIdx.y] = 0;
// ....else
// ........data[threadIdx.x][threadIdx.y] = d_image[ gLoc - KERNEL_RADIUS - IMUL(dataW, KERNEL_RADIUS)];

// ....// case2: upper right
// ....x = x0 + KERNEL_RADIUS;
// ....y = y0 - KERNEL_RADIUS;
// ....if ( x > dataW-1 || y < 0 )
// ........data[threadIdx.x + blockDim.x][threadIdx.y] = 0;
// ....else
// ........data[threadIdx.x + blockDim.x][threadIdx.y] = d_Data[gLoc + KERNEL_RADIUS - IMUL(dataW, KERNEL_RADIUS)];

// ....// case3: lower left
// ....x = x0 - KERNEL_RADIUS;
// ....y = y0 + KERNEL_RADIUS;
// ....if (x < 0 || y > dataH-1)
// ........data[threadIdx.x][threadIdx.y + blockDim.y] = 0;
// ....else
// ........data[threadIdx.x][threadIdx.y + blockDim.y] = d_Data[gLoc - KERNEL_RADIUS + IMUL(dataW, KERNEL_RADIUS)];

// ....// case4: lower right
// ....x = x0 + KERNEL_RADIUS;
// ....y = y0 + KERNEL_RADIUS;
// ....if ( x > dataW-1 || y > dataH-1)
// ........data[threadIdx.x + blockDim.x][threadIdx.y + blockDim.y] = 0;
// ....else
// ........data[threadIdx.x + blockDim.x][threadIdx.y + blockDim.y] = d_Data[gLoc + KERNEL_RADIUS + IMUL(dataW, KERNEL_RADIUS)];

// ....__syncthreads();

// ....// convolution
// ....float sum = 0;
// ....x = KERNEL_RADIUS + threadIdx.x;
// ....y = KERNEL_RADIUS + threadIdx.y;
// ....for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++)
// ........for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
// ............sum += data[x + i][y + j] * d_Kernel[KERNEL_RADIUS + j] * d_Kernel[KERNEL_RADIUS + i];

// ....d_Result[gLoc] = sum;
}

#endif
