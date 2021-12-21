#ifndef _GEMM_H
#define _GEMM_H
#include <stdio.h>
#include <cuda_runtime.h>

typedef struct{
    int width;
    int height;
    float* elements;
} Mat;
const int TILE_SIZE = 16;
const int VECTOR_SIZE = 4;
int C_ORDER = 0;
int FORTRAN_ORDER = 1;

void initdata(float*, const int);
void cleardata(Mat*);
void dispMat(Mat* );
bool checkResult(Mat*, Mat *, int);
void MatMulHost(Mat* , Mat* , Mat* );

__global__ void warmup(Mat* , Mat* , Mat* );
__global__ void MatMulKernel(Mat* , Mat* , Mat* );
__global__ void MatMulKernel_Tiling(Mat *, Mat *, Mat *);
__global__ void MatMulKernel_Tiling_Coalesing(Mat *, Mat *, Mat *);
__global__ void MatMulKernel_Tiling_noBankflict(Mat *, Mat *, Mat *);
__global__ void MatMulKernel_Tiling_outProd(Mat *, Mat *, Mat *);
__global__ void MatMulKernel_1x4(Mat* , Mat* , Mat* );
__global__ void MatMulKernel_4x4(Mat* , Mat* , Mat* );

#endif