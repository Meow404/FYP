#ifndef IMAGECONVOLUTIONPARALLELSHAREDCONSTANTMEMORYNOOVERLAP
#define IMAGECONVOLUTIONPARALLELSHAREDCONSTANTMEMORYNOOVERLAP
#define KERNEL_OFFSET 3
// #define BLOCK_WIDTH 13

float *applyKernelToImageParallelSharedConstantMemoryNoOverlap(float *image, int imageWidth, int imageHeight, kernel kernel, char *imagePath, int blockWidth);
// float applyKernelPerPixelSharedMemory(int y, int x, int kernelX, int kernelY, int imageWidth, int imageHeight, float *kernel, float *image);
__global__ void applyKernelPerPixelParallelSharedConstantMemoryNoOverlap(float *d_image, float *d_sumArray);

float *applyKernelToImageParallelSharedConstantMemoryNoOverlap(float *image, int imageWidth, int imageHeight, kernel kernel, char *imagePath, int blockWidth)
{
  //printImage(image, imageWidth, imageHeight, "orginalImagePartition.txt");
  float *d_image, *d_sumArray;

  int sizeInt = sizeof(int);
  int sizeFloat = sizeof(float);
  int sizeImageArray = imageWidth * imageHeight * sizeFloat;
  float *sumArray = (float *)malloc(sizeImageArray);

  cudaMalloc((void **)&d_image, sizeImageArray);
  cudaMalloc((void **)&d_sumArray, sizeImageArray);

  cudaMemcpy(d_image, image, sizeImageArray, cudaMemcpyHostToDevice);

  //constants
  cudaMemcpyToSymbol(kernelConstant, kernel.matrix, sizeof(float) * kernel.dimension * kernel.dimension);
  cudaMemcpyToSymbol(imageWidthConstant, &imageWidth, sizeInt);
  cudaMemcpyToSymbol(imageHeightConstant, &imageHeight, sizeInt);
  cudaMemcpyToSymbol(kernelDimensionXConstant, &kernel.dimension, sizeInt);
  cudaMemcpyToSymbol(kernelDimensionYConstant, &kernel.dimension, sizeInt);

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
  applyKernelPerPixelParallelSharedConstantMemoryNoOverlap<<<dimGrid, dimBlock, (blockWidth + kernel.dimension - 1) * (blockWidth + kernel.dimension - 1) * sizeFloat>>>(d_image, d_sumArray);

  cudaError_t errSync = cudaGetLastError();
  cudaError_t errAsync = cudaDeviceSynchronize();
  if (errSync != cudaSuccess)
    printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
  if (errAsync != cudaSuccess)
    printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));

  cudaMemcpy(sumArray, d_sumArray, sizeImageArray, cudaMemcpyDeviceToHost);

  // CUDA free varibles
  cudaFree(d_image);
  cudaFree(d_sumArray);

  return sumArray;
}

__global__ void applyKernelPerPixelParallelSharedConstantMemoryNoOverlap(float *d_image, float *d_sumArray)
{
  // int comp = 45;
  int offsetX = (kernelDimensionXConstant - 1) / 2;
  int offsetY = (kernelDimensionYConstant - 1) / 2;

  int memDimX = blockDim.x + kernelDimensionXConstant - 1;
  int memDimY = blockDim.y + kernelDimensionYConstant - 1;

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

  if ((x - offsetX) < 0 || (y - offsetY) < 0)
    local_imageSection[row * (memDimX) + col] = 0;
  else
    local_imageSection[row * (memDimX) + col] = d_image[(y - offsetY) * (imageWidthConstant) + (x - offsetX)];

  if (row + blockDim.y < memDimY && col + blockDim.x < memDimX)
    if (x - offsetX + blockDim.x >= imageWidthConstant || y - offsetY + blockDim.y >= imageHeightConstant)
      local_imageSection[(row + blockDim.y) * (memDimX) + (col + blockDim.x)] = 0;
    else
      local_imageSection[(row + blockDim.y) * (memDimX) + (col + blockDim.x)] = d_image[(y - offsetY + blockDim.y) * (imageWidthConstant) + (x - offsetX + blockDim.x)];

  if (row + blockDim.y < memDimY)
    if (x - offsetX < 0 || y - offsetY + blockDim.y >= imageHeightConstant)
      local_imageSection[(row + blockDim.y) * (memDimX) + col] = 0;
    else
      local_imageSection[(row + blockDim.y) * (memDimX) + col] = d_image[(y - offsetY + blockDim.y) * (imageWidthConstant) + (x - offsetX)];

  if (col + blockDim.x < memDimX)
    if (x - offsetX + blockDim.x >= imageWidthConstant || y - offsetY < 0)
      local_imageSection[row * (memDimX) + (col + blockDim.x)] = 0;
    else
      local_imageSection[row * (memDimX) + (col + blockDim.x)] = d_image[(y - offsetY) * (imageWidthConstant) + (x - offsetX + blockDim.x)];

  __syncthreads();

  // convolution
     if (y < (imageHeightConstant) && x < (imageWidthConstant))
  {
  float sum = 0.0;
  for (int j = 0; j < kernelDimensionYConstant; j++)
  {
    for (int i = 0; i < kernelDimensionXConstant; i++)
    {

      float value;

      float k = kernelConstant[i + j * (kernelDimensionYConstant)];
      float imageElement = local_imageSection[(row + j) * (memDimX) + col + i];

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
#endif
