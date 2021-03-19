#ifndef IMAGECONVOLUTIONSERIAL
#define IMAGECONVOLUTIONSERIAL
// #define KERNELDIMENSION 3

float *applyKernelToImageSerial(float *image, int imageWidth, int imageHeight, kernel kernel, char *imagePath);
float applyKernelPerPixel(int y, int x, int kernelX, int kernelY, int imageWidth, int imageHeight, float *kernel, float *image);

float *applyKernelToImageSerial(float *image, int imageWidth, int imageHeight, kernel kernel, char *imagePath)
{
  //printImage(image,imageWidth,imageHeight,"originalImage.txt");
  unsigned int size = imageWidth * imageHeight * sizeof(float);
  float *newImage = (float *)malloc(size);
  for (int y = 0; y < imageHeight; y++)
  {
    for (int x = 0; x < imageWidth; x++)
    {
      float sum = applyKernelPerPixel(y, x, kernel.dimension, kernel.dimension, imageWidth, imageHeight, kernel.matrix, image);
      //Normalising output
      // if (sum < 0)
      //   sum = 0;
      // else if (sum > 1)
      //   sum = 1;
      newImage[y * imageWidth + x] = sum;
    }
  }

  return newImage;
}

float applyKernelPerPixel(int y, int x, int kernelX, int kernelY, int imageWidth, int imageHeight, float *kernel, float *image)
{
  /*
This function applies the kernel on the pixal @ (x,y) of the image.

parameters:
int y : Y-coordinate of pixel
int x : X-coordinate of pixel
int kernelX : Kernel width
int kernelY : Kernel height 
int imageWidth : Image width
int imageHeight : Image Height
float *kernel : 2D float matrix containing kernel values
float *image : 2D float matrix containing image values

return : value of convoluted image pixel @ (x,y)
*/
  float sum = 0;
  int offsetX = (kernelX - 1) / 2;
  int offsetY = (kernelY - 1) / 2;

  for (int j = 0; j < kernelY; j++)
  {
    //Ignore out of bounds
    if (y + j < offsetY || y + j - offsetY >= imageHeight)
      continue;

    for (int i = 0; i < kernelX; i++)
    {
      //Ignore out of bounds
      if (x + i < offsetX || x + i - offsetX >= imageWidth)
        continue;

      float k = kernel[i + j * kernelY];                                                         //Get value at kernel positions (i,j)
      float imageElement = image[y * imageWidth + imageWidth * (j - offsetY) + x + i - offsetX]; //Get value at image positions (x - offset + i, y - offset + j)
      float value = k * imageElement;
      sum = sum + value;
    }
  }
  return sum;
}

#endif
