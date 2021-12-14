#include <stdio.h>
typedef struct{
    int width;
    int height;
    float* elements;
} Mat;

void initdata(float* A, const int size)
{
    for(int i = 0; i < size; ++i){
        A[i] = (float)(rand()&0xFF);
    }
    return;
}
void cleardata(float* A, const int size)
{
    for(int i = 0; i < size; ++i){
        A[i] = (float)(0);
    }
    return;
}

void dispMat(Mat* A){
    for(int ix = 0; ix < A->height; ++ix){
        for(int iy = 0; iy < A->width; ++iy){
            printf("%f ",A->elements[ix * A->width + iy]);
        }
        printf("\n");
    }
    printf("\n");
}

void checkResult(Mat* hostRef, Mat *gpuRef)
{
    double epsilon = 1.0E-3;
    bool match = 1;

    for (int i = 0; i < hostRef->width * hostRef->height; i++)
    {
        if (abs(hostRef->elements[i] - gpuRef->elements[i]) > epsilon)
        {
            match = 0;
            printf("host %f gpu %f\n", hostRef->elements[i], gpuRef->elements[i]);
            break;
        }
    }

    if (!match)
    {
        printf("------------Arrays NOT match--------------------.\n\n");
    }
}

void MatMulHost(Mat* A, Mat* B, Mat* C)
{
    for(int i = 0; i < C->height; ++ i){
        for(int j = 0; j < C->width; ++j){
            C->elements[i * C->width + j] = 0;
            for(int k = 0; k < B->height; ++k){
                C->elements[i * C->width + j] += A->elements[i * A->width + k] * B->elements[k * B->width + j];
            }
        }
    }
}