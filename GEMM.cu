//matrix multiple (nx*nk)(nk*ny) = (nx*ny)

#include "include/common.h"
#include <cuda_runtime.h>
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

    if (match)
    {
        printf("Arrays match.\n\n");
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

__global__ void MatMulKernel_1x4(Mat* A, Mat* B, Mat* C)
{
    unsigned int row = blockIdx.y * blockDim.y  + threadIdx.y;	
	unsigned int col = blockIdx.x * blockDim.x * 4  + threadIdx.x;
    float tmp1 = 0, tmp2 = 0, tmp3 = 0, tmp = 0;
    // float a;
    if (row < C->height && col + 3 < C->width){
        for(int ik = 0; ik < A->width; ++ik){
            // a = A->elements[row * A->width + ik];
            tmp += A->elements[row * A->width + ik] * B->elements[ik * B->width + col];
            tmp1 += A->elements[row * A->width + ik] *B->elements[ik * B->width + col + 1]; // C[x][y+1] = a[x][k] * b[k][y+1]
            tmp2 += A->elements[row * A->width + ik] *B->elements[ik * B->width + col + 2]; // C[x][y+2] = a[x][k] * b[k][y+2]
            tmp3 += A->elements[row * A->width + ik] *B->elements[ik * B->width + col + 3]; // C[x][y+3] = a[x][k] * b[k][y+3]

        }
        C->elements[row * C->width + col] = tmp;
        C->elements[row * C->width + col+1] = tmp1;
        C->elements[row * C->width + col+2] = tmp2;
        C->elements[row * C->width + col+3] = tmp3;
    }
}

__global__ void MatMulKernel_4x4(Mat* A, Mat* B, Mat* C)
{
    unsigned int row = (blockIdx.y * blockDim.y * 4  + threadIdx.y) ;	
	unsigned int col = (blockIdx.x * blockDim.x * 4 + threadIdx.x);
    float tmp1, tmp2, tmp3, tmp;
    // float a;
    if (row < C->height && col < C->width){
        for(int i = 0; i < 4; ++i){
            tmp1 = 0, tmp2 = 0, tmp3 = 0, tmp = 0;
            for(int ik = 0; ik < A->width; ++ik){
            // a = A->elements[row * A->width + ik];
                
                tmp += A->elements[row * A->width + ik] * B->elements[ik * B->width + col];
                tmp1 += A->elements[row * A->width + ik] *B->elements[ik * B->width + col + 1]; // C[x][y+1] = a[x][k] * b[k][y+1]
                tmp2 += A->elements[row * A->width + ik] *B->elements[ik * B->width + col + 2]; // C[x][y+2] = a[x][k] * b[k][y+2]
                tmp3 += A->elements[row * A->width + ik] *B->elements[ik * B->width + col + 3]; // C[x][y+3] = a[x][k] * b[k][y+3]
            }
            C->elements[row * C->width + col] = tmp;
            C->elements[row * C->width + col+1] = tmp1;
            C->elements[row * C->width + col+2] = tmp2;
            C->elements[row * C->width + col+3] = tmp3;
            ++ row;
        }
        
    }
}

int main(){
    /*
    A: X * K
    B: K * Y
    C, D: X * Y
    */
    Mat *A, *B, *kernel1, * kernel2;
    Mat *Host = (Mat*)malloc(sizeof(Mat));
    CHECK(cudaMallocManaged((float **)&A, sizeof(Mat)));
    CHECK(cudaMallocManaged((float **)&B, sizeof(Mat)));
    CHECK(cudaMallocManaged((float **)&kernel1, sizeof(Mat)));
    CHECK(cudaMallocManaged((float **)&kernel2, sizeof(Mat)));
    int nshift = 4;
    int nx, ny, nk;
    nx = ny = nk = 1<<nshift;
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

    // CHECK(cudaMallocManaged((float **)&kernel2, nxy * sizeof(float)));
    Host->elements = (float *)malloc(nBytes);

    double iStart = seconds();
    initdata(A->elements, nxk);
    initdata(B->elements, nky);
    double iElaps = seconds() - iStart;
    printf("initialization: \t%f\n", iElaps);

    iStart = seconds();
    MatMulHost(A, B, Host);
    iElaps = seconds() - iStart;
    printf("mulMat on host: \t%f\n", iElaps);
    // dispMat(A);
    // dispMat(Host);

    // Naive MatMul on GPU
    int block_x = 4;
    int block_y = 4;
    dim3 block(block_x, block_y);
    dim3 grid((nx + block.x - 1)/block.x, (ny + block.y - 1)/block.y);
    iStart = seconds();
    MatMulKernel<<<grid, block>>>(A, B, kernel1);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("mulMat on device orgin: \t%f\n", iElaps);
    CHECK(cudaGetLastError());
    checkResult(kernel1, Host);
    dispMat(kernel1);


    iStart = seconds();
    dim3 block1(block.x/4,block.y);
    printf("%d %d %d %d\n",grid.x, grid.y, block1.x, block1.y);
    MatMulKernel_1x4<<<grid, block1>>>(A, B, kernel2);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("mulMat on device 1*4: \t%f\n", iElaps);
    CHECK(cudaGetLastError());
    checkResult(Host, kernel2);
    dispMat(kernel2);
    
    // cleardata(kernel1->elements, nxy);
    // iStart = seconds();
    // dim3 grid2(grid.x/4,grid.y/4);
    // MatMulKernel_4x4<<<grid2, block>>>(A, B, kernel1);
    // CHECK(cudaDeviceSynchronize());
    // iElaps = seconds() - iStart;
    // printf("mulMat on device 1*4: \t%f\n", iElaps);
    // CHECK(cudaGetLastError());
    // checkResult(Host, kernel1);
    return 0;
}