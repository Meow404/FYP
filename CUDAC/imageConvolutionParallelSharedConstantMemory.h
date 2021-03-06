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

float *applyKernelToImageParallelSharedConstantMemory(float *image, int imageWidth, int imageHeight, kernel kernel, char *imagePath, int blockWidth)
{
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

  int overlapX = (kernel.dimension - 1);
  int overlapY = (kernel.dimension - 1);

  int numHorBlocks = (imageWidth) / (blockWidth - overlapX);
  int numVerBlocks = (imageHeight) / (blockWidth - overlapY);

  if (imageWidth % (blockWidth - overlapX))
    numHorBlocks++;
  if (imageHeight % (blockWidth - overlapY))
    numVerBlocks++;

  dim3 dimGrid(numVerBlocks, numHorBlocks, 1);
  dim3 dimBlock(blockWidth, blockWidth, 1);
  applyKernelPerPixelParallelSharedConstantMemory<<<dimGrid, dimBlock, blockWidth*blockWidth*sizeFloat>>>(d_image, d_sumArray);
  cudaMemcpy(sumArray, d_sumArray, sizeImageArray, cudaMemcpyDeviceToHost);

  // CUDA free varibles
  cudaFree(d_image);
  cudaFree(d_sumArray);

  return sumArray;
}
__global__ void applyKernelPerPixelParallelSharedConstantMemory(float *d_image, float *d_sumArray)
{
  // int comp = 45;
  int offsetX = (kernelDimensionXConstant - 1) / 2;
  int offsetY = (kernelDimensionYConstant - 1) / 2;

  int overlapX = (kernelDimensionXConstant - 1);
  int overlapY = (kernelDimensionYConstant - 1);

  int y = blockIdx.y * (blockDim.y - overlapY ) + threadIdx.y;
  int x = blockIdx.x * (blockDim.x - overlapX) + threadIdx.x;
  // int y = blockIdx.y * blockDim.y + threadIdx.y;
  // int x = blockIdx.x * blockDim.x + threadIdx.x;

  int row = threadIdx.y;
  int col = threadIdx.x;

  extern __shared__ float local_imageSection[];
  // int imageIndex = y * (imageWidthConstant) + x;
  // local_imageSection[row][col] = d_image[y * (*d_imageWidth) + x - 2 * blockIdx.x];
  local_imageSection[row*(blockDim.x) + col] = d_image[y * (imageWidthConstant) + x];

  __syncthreads();

  //Need to fill in if statement ******
  if ((blockIdx.x == 0 || (blockIdx.x != 0 && threadIdx.x >= offsetX)) && threadIdx.x < blockDim.x - offsetX && (blockIdx.y == 0 || (blockIdx.y != 0 && threadIdx.y >= offsetY)) && threadIdx.y < blockDim.y - offsetY) 
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
      d_sumArray[y * (imageWidthConstant) + x] = sum;
    }
  }
}

#endif
