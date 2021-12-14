//matrix multiple (nx*nk)(nk*ny) = (nx*ny)

#include "include/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

#include "src/utils.c"

const int TILE_SIZE = 16;

__global__ void warmup(Mat* A, Mat* B, Mat* C)
{
	// Each thread computes one element of C
	// by accumulating results into Cvalue
	float Cvalue = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;	//结果矩阵C的行索引
	int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < C->height && col < C->width){	//结果矩阵C的列索引
        for (int e = 0; e < A->width; ++e)
        {
            Cvalue += A->elements[row * A->width + e]			//所有点到点的元素乘积求和
                    * B->elements[e * B->width + col];
        }
	    C->elements[row * C->width + col] = Cvalue;

    }
}


__global__ void MatMulKernel(Mat* A, Mat* B, Mat* C)
{
	// Each thread computes one element of C
	// by accumulating results into Cvalue
	float Cvalue = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;	//结果矩阵C的行索引
	int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < C->height && col < C->width){	//结果矩阵C的列索引
        for (int e = 0; e < A->width; ++e)
        {
            Cvalue += A->elements[row * A->width + e]			//所有点到点的元素乘积求和
                    * B->elements[e * B->width + col];
        }
	    C->elements[row * C->width + col] = Cvalue;

    }
}

int main(int argc, char *argv[]){
    /*
    A: X * K
    B: K * Y
    C, D: X * Y
    */
    Mat *A, *B, *kernel1, *kernel2;
    Mat *Host = (Mat*)malloc(sizeof(Mat));
    CHECK(cudaMallocManaged((float **)&A, sizeof(Mat)));
    CHECK(cudaMallocManaged((float **)&B, sizeof(Mat)));
    CHECK(cudaMallocManaged((float **)&kernel1, sizeof(Mat)));
    CHECK(cudaMallocManaged((float **)&kernel2, sizeof(Mat)));

    
    int nshift = 512;
    if (argc > 1){nshift = atoi(argv[1]);}

    int nx, ny, nk;
    nx = ny = nk = nshift;
    A->height = nx;A->width = nk;
    B->height = nk;B->width = ny;
    Host->height = nx;Host->width = ny;
    kernel1->height = nx;kernel1->width = ny;
    kernel2->height = nx;kernel2->width = ny;

    int nxk = nx * nk;
    int nky = nk * ny;
    int nxy = nx *ny;
    int nBytes = sizeof(float) * nxy;
    CHECK(cudaMallocManaged((float **)&A->elements, nxk * sizeof(float)));
    CHECK(cudaMallocManaged((float **)&B->elements, nky * sizeof(float)));
    CHECK(cudaMallocManaged((float **)&kernel1->elements, nxy * sizeof(float)));
    CHECK(cudaMallocManaged((float **)&kernel2->elements, nxy * sizeof(float)));


    Host->elements = (float *)malloc(nBytes);

    double iStart = seconds();
    initdata(A->elements, nxk);
    initdata(B->elements, nky);
    double iElaps = seconds() - iStart;
    // printf("initialization: \t%f\n", iElaps);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((nx + block.x - 1)/block.x, (ny + block.y - 1)/block.y);
    iStart = seconds();
    warmup<<<grid, block>>>(A, B, kernel1);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    //printf("mulMat on warmup: \t%f\n", iElaps);

    iStart = seconds();
    MatMulKernel<<<grid, block>>>(A, B, kernel2);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("mulMat on device orgin: \t%f\n", iElaps);
    CHECK(cudaGetLastError());
    checkResult(kernel2, kernel1);
    return 0;
}