#ifndef IMAGECONVOLUTIONPARALLELCONSTANTMEMORY
#define IMAGECONVOLUTIONPARALLELCONSTANTMEMORY

float *applyKernelToImageParallelConstantMemory(float *image, int imageWidth, int imageHeight, kernel kernel, char *imagePath, int blockWidth);
// float applyKernelPerPixelConstantMemory(int y, int x, int kernelX, int kernelY, int imageWidth, int imageHeight, float *kernel, float *image);
__global__ void applyKernelPerPixelParallelConstantMemory(float *d_image, float *d_sumArray);

__constant__ float kernelConstant[10 * 10];
__constant__ int imageWidthConstant;
__constant__ int imageHeightConstant;
__constant__ int kernelDimensionXConstant;
__constant__ int kernelDimensionYConstant;

float *applyKernelToImageParallelConstantMemory(float *image, int imageWidth, int imageHeight, kernel kernel, char *imagePath, int blockWidth)
{
    float *d_kernel, *d_image, *d_sumArray;

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

    int numHorBlocks = imageWidth / blockWidth;
    int numVerBlocks = imageHeight / blockWidth;

    if (imageWidth % blockWidth)
        numHorBlocks++;
    if (imageHeight % blockWidth)
        numVerBlocks++;

    dim3 dimGrid(numVerBlocks, numHorBlocks, 1);
    dim3 dimBlock(blockWidth, blockWidth, 1);

    applyKernelPerPixelParallelConstantMemory<<<dimGrid, dimBlock>>>(d_image, d_sumArray);
    cudaMemcpy(sumArray, d_sumArray, sizeImageArray, cudaMemcpyDeviceToHost);

    // CUDA free varibles
    cudaFree(d_image);
    cudaFree(d_sumArray);

    return sumArray;
}
__global__ void applyKernelPerPixelParallelConstantMemory(float *d_image, float *d_sumArray)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetX = (kernelDimensionXConstant - 1) / 2;
    int offsetY = (kernelDimensionYConstant - 1) / 2;
    float sum = 0.0;
    if ((y < (imageHeightConstant)) && (x < (imageWidthConstant)))
    {
        for (int j = 0; j < kernelDimensionYConstant; j++)
        {
            //Ignore out of bounds
            if (y + j < offsetY || y + j - offsetY >= imageHeightConstant)
                continue;

            for (int i = 0; i < kernelDimensionXConstant; i++)
            {
                //Ignore out of bounds
                if (x + i < offsetX || x + i - offsetX >= imageWidthConstant)
                    continue;

                float k = kernelConstant[i + j * (kernelDimensionYConstant)];
                float imageElement = d_image[y * (imageHeightConstant) + x + i - offsetX + (imageWidthConstant) * (j - 1)];

                float value = k * imageElement;
                sum = sum + value;
            }
        }
        // int imageIndex = y * (imageHeightConstant) + x;

        // if (sum < 0)
        //     sum = 0;
        // else if (sum > 1)
        //     sum = 1;

        d_sumArray[y * (imageHeightConstant) + x] = sum;
    }
}
#endif
