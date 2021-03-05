#include "imageHandler.h"
#include "kernelHandler.h"
// #ifndef IMAGECONVOLUTIONSERIAL
// #define IMAGECONVOLUTIONSERIAL
// #define KERNELDIMENSION 3

void applyKernelToImageSerial(float *image, int imageWidth, int imageHeight, float *kernel, int kernelDimension, char *imagePath);
void flipKernel(float *kernel, int kernelDimension);
void loadAllKernels(float **kernels, FILE *fp);
int getNumKernels(FILE *fp);
float applyKernelPerPixel(int y, int x, int kernelX, int kernelY, int imageWidth, int imageHeight, float *kernel, float *image);

class imageConvolution
{

public:
    void convolution(const char *imagePath, const char *kernelPath)
    {

        imageHandler image = imageHandler(imagePath);
        kernelHandler kernels = kernelHandler(kernelPath);

        for (int i = 0; i < 10; i++)
        {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);

            for (int i = 0; i < kernels.getNumOfKernels(); i++)
            {
                applyKernelToImage(image, kernels.getKernel(i), imagePath);
            }

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            printf("Time Serial Implementation: %f \n", milliseconds);
        }
    }

    virtual void applyKernelToImage(imageHandler image, kernel k, const char *imagePath) = 0;
    // virtual float applyKernelPerPixel(int x, int y, kernel k, float *image, int imageWidth, int imageHeight) = 0;
};
// #endif

class serialConvolution : public imageConvolution
{
public:
    void applyKernelToImage(imageHandler image, kernel k, const char *imagePath)
    {
        //printImage(image,imageWidth,imageHeight,"originalImage.txt");
        float *newImage = new float[image.getWidth() * image.getHeight()];
        for (int y = 0; y < image.getHeight(); y++)
        {
            for (int x = 0; x < image.getWidth(); x++)
            {
                float sum = applyKernelPerPixel(x, y, k, image.get1DImageArray(), image.getWidth(), image.getHeight());
                //Normalising output
                if (sum < 0)
                    sum = 0;
                else if (sum > 1)
                    sum = 1;
                newImage[y * image.getWidth() + x] = sum;
            }
        }

        imageHandler outputImage = imageHandler(image.getWidth(), image.getHeight(), newImage);
        char outputFilename[1024];
        strcpy(outputFilename, imagePath);
        strcpy(outputFilename + strlen(imagePath) - 4, "_serial_out.pgm");
        outputImage.saveImage(outputFilename);
    }

    float applyKernelPerPixel(int x, int y, kernel k, float *image, int imageWidth, int imageHeight)
    {
        float sum = 0;
        int offset = (k.dimension - 1) / 2;

        for (int j = 0; j < k.dimension; j++)
        {
            //Ignore out of bounds
            if (y + j < offset || y + j - offset >= imageHeight)
                continue;

            for (int i = 0; i < k.dimension; i++)
            {
                //Ignore out of bounds
                if (x + i < offset || x + i - offset >= imageWidth)
                    continue;

                float p_value = k.matrix[i][j];
                float imageElement = image[y * imageWidth + x + i - offset + imageWidth * (j - 1)];
                float value = p_value * imageElement;
                sum = sum + value;
            }
        }
        return sum;
    }
};

class parallelConvolution : public imageConvolution
{
public:
    void applyKernelToImage(imageHandler image, kernel k, const char *imagePath)
    {
        int *d_kernelDimensionX, *d_kernelDimensionY, *d_imageWidth, *d_imageHeight;
        float *d_kernel, *d_image, *d_sumArray;

        int sizeInt = sizeof(int);
        int sizeFloat = sizeof(float);
        float *newImage = new float[image.getWidth() * image.getHeight()];

        cudaMalloc((void **)&d_kernelDimensionX, sizeInt);
        cudaMalloc((void **)&d_kernelDimensionY, sizeInt);
        cudaMalloc((void **)&d_imageWidth, sizeInt);
        cudaMalloc((void **)&d_imageHeight, sizeInt);
        cudaMalloc((void **)&d_kernel, k.dimension * k.dimension * sizeFloat);
        cudaMalloc((void **)&d_image, image.getHeight()*image.getWidth());
        cudaMalloc((void **)&d_sumArray, image.getHeight()*image.getWidth());

        cudaMemcpy(d_kernelDimensionX, &k.dimension, sizeInt, cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernelDimensionY, &k.dimension, sizeInt, cudaMemcpyHostToDevice);
        cudaMemcpy(d_imageWidth, image.getWidth(), sizeInt, cudaMemcpyHostToDevice);
        cudaMemcpy(d_imageHeight, image.getHeight(), sizeInt, cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernel, k.matrix, k.dimension * k.dimension * sizeFloat, cudaMemcpyHostToDevice);
        cudaMemcpy(d_image, image, image.getWidth() * image.getHeight() * sizeFloat, cudaMemcpyHostToDevice);

        int width = imageWidth * imageHeight;
        int numBlocks = (imageWidth) / BLOCK_WIDTH;
        if (width % BLOCK_WIDTH)
            numBlocks++;
        dim3 dimGrid(numBlocks, numBlocks, 1);
        dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
        
        applyKernelPerPixel<<<dimGrid, dimBlock>>>(d_kernelDimensionX, d_kernelDimensionY, d_imageWidth, d_imageHeight, d_kernel, d_image, d_sumArray);
        
        cudaMemcpy(sumArray, d_sumArray, sizeImageArray, cudaMemcpyDeviceToHost);

        //printImage(sumArray,imageWidth,imageHeight,"newImageP.txt");
        char outputFilename[1024];
        strcpy(outputFilename, imagePath);
        strcpy(outputFilename + strlen(imagePath) - 4, "_parallel_out.pgm");
        sdkSavePGM(outputFilename, sumArray, imageWidth, imageHeight);
    }
    __global__ void applyKernelPerPixel(int *d_kernelDimensionX, int *d_kernelDimensionY, int *d_imageWidth, int *d_imageHeight, float *d_kernel, float *d_image, float *d_sumArray)
    {

        int comp = 45;
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
            //Normalising output
            if (sum < 0)
                sum = 0;
            else if (sum > 1)
                sum = 1;
            d_sumArray[y * (*d_imageWidth) + x] = sum;
        }
    }
};