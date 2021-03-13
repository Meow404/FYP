#ifndef _helper_h   
#define _helper_h

#include <iostream>
#include <stdio.h>

void flipKernel(float* kernel, int kernelDimension);
void loadKernels(float * kernel, char buf[512]);
void loadAllKernels(float ** kernels,  FILE* fp);
int getNumKernels(FILE* fp);
void loadRow(float * matrix, int row, int kernel_dim, char buf[512]);

struct kernel
{
    int dimension;
    float *matrix;
};

kernel** loadAllKernels(FILE* fp, int numOfKernels){   
    char buf[512];
    int index = 0;

    kernel **kernels = (kernel **)malloc(sizeof(kernel *) * numOfKernels);

    for (int i = 0; i < numOfKernels; i++)
    {
        int kernel_dim;
        fgets(buf, sizeof(buf), fp);
        sscanf(buf, "%d", &kernel_dim);
        printf("Loading kernel with kernel dimesnion %dx%d\n", kernel_dim, kernel_dim);

        kernel kl;
        kl.dimension = kernel_dim;
        kl.matrix = (float *)malloc(sizeof(float) * kernel_dim * kernel_dim);

        for (int j = 0; j < kernel_dim; j++)
        {
            // kl.matrix[j] = new float[kernel_dim];
            fgets(buf, sizeof(buf), fp);

            loadRow(kl.matrix, j, kernel_dim, buf);
        }
        kernels[i] = (kernel *)malloc(sizeof(kernel));
        kernels[i] = &kl;
    }

    printf("\nNum of kernels : %d", numOfKernels);
    for (int k = 0; k < numOfKernels; k++)
    {
        printf("\n%dx%d", kernels[k]->dimension, kernels[k]->dimension);
        for (int i = 0; i < kernels[k]->dimension; i++)
        {
            printf("\n");
            for (int j = 0; j < kernels[k]->dimension; j++)
            {
                printf("%f ", kernels[k]->matrix[i*kernels[k]->dimension + j]);
            }
        }
        std::cout << "\n====================================";
    }
    return kernels;
   
    // while (fgets(buf, sizeof(buf), fp) != NULL)
    // {
    //   loadKernels(kernels[index],buf);
    //   index++;
    // }
      
}

void loadRow(float * matrix, int row, int kernel_dim, char buf[512]){ 
    int count = 0;
    buf[strlen(buf) - 1] = '\0';
    const char delimeter[2] = ",";
    char* token = strtok(buf, delimeter);
    while( token != NULL){
      sscanf(token, "%f,", &matrix[row*kernel_dim + count] );
      token = strtok(NULL,delimeter);
      count = count + 1;
        
    } 
}

//https://www.geeksforgeeks.org/write-a-program-to-reverse-an-array-or-string/
void flipKernel(float* kernel, int kernelDimension){
  int temp;
  int start = 0;
  int end = kernelDimension * kernelDimension - 1;
  while (start < end)
  {
      temp = kernel[start];   
      kernel[start] = kernel[end];
      kernel[end] = temp;
      start++;
      end--;
  }   
      
}

void printImage(float *image,int width,int height,char * fileName){
   FILE *f = fopen(fileName, "w");
   if (f == NULL)
   {
    printf("Error opening file!\n");
    exit(1);
   }

   for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            fprintf(f,"%f ", image[i*width+j]);
            if((i*width+j) % 12 == 0){
               fprintf(f,"\n");
           }
        }
        fprintf(f,"\n");
    }
}

void printKernel(float* kernel, int kernelDimension){
  for (int i = 0; i < kernelDimension; i++) {
       for (int j = 0; j < kernelDimension; j++) {
           printf("%f ", kernel[i*kernelDimension+j]);
       }
       printf("\n");
   }
}

int getNumKernels(FILE* fp){
    return 1;
//   int ch, lines=0;
//   while(!feof(fp))
//   {
//        printf("2");
//       ch = fgetc(fp);
//       if(ch == '\n')
//       {
//         lines++;
//       }
//   }
//   rewind(fp);
//   return lines;
}
#endif 