#include <cuda_runtime.h>
#include <stdio.h>
#include <cublas_v2.h>

#define TILE_SIZE 16
#define VECTOR_SIZE 4
#define KERNEL_NUM 9

void test_cublas(Mat* A, Mat* B, Mat* C){
    CHECK(cudaDeviceSynchronize());
    cublasHandle_t handle;
    cublasCreate(&handle);
    float al=1, bet=0;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, A->height, C->width, A->width, 
		    &al, A->elements, A->height, B->elements, B->height, &bet, C->elements, C->height);
}
void test_mulMat_naive(Mat* A, Mat* B, Mat* C, dim3 block, dim3 grid){
    CHECK(cudaDeviceSynchronize());
    MatMulKernel<<<grid, block>>>(A, B, C);
}
void test_mulMat_1x4(Mat* A, Mat* B, Mat* C, dim3 block, dim3 grid){
    CHECK(cudaDeviceSynchronize());
    dim3 block1(block.x/4, block.y);
    MatMulKernel_1x4<<<grid, block1>>>(A, B, C);
}
void test_mulMat_4x4(Mat* A, Mat* B, Mat* C, dim3 block, dim3 grid){
    CHECK(cudaDeviceSynchronize());
    dim3 block1(block.x/4, block.y/4);
    MatMulKernel_4x4<<<grid, block1>>>(A, B, C);
}
void test_mulMat_Tiling(Mat* A, Mat* B, Mat* C, dim3 block, dim3 grid){
    CHECK(cudaDeviceSynchronize());
    MatMulKernel_Tiling<<<grid, block>>>(A, B, C);
}
void test_mulMat_Tiling_Coalesing(Mat* A, Mat* B, Mat* C, dim3 block, dim3 grid){
    CHECK(cudaDeviceSynchronize());
    MatMulKernel_Tiling_Coalesing<<<grid, block>>>(A, B, C);
}
void test_mulMat_Tiling_noBankconflict(Mat* A, Mat* B, Mat* C, dim3 block, dim3 grid){
    CHECK(cudaDeviceSynchronize());
    MatMulKernel_Tiling_noBankflict<<<grid, block>>>(A, B, C);
}
void test_mulMat_outerproduct(Mat* A, Mat* B, Mat* C, dim3 block, dim3 grid){
    CHECK(cudaDeviceSynchronize());
    dim3 block1(TILE_SIZE, VECTOR_SIZE);
    dim3 grid1((C->width + block.x - 1)/(TILE_SIZE*VECTOR_SIZE), (C->height + block.y - 1)/TILE_SIZE);
    MatMulKernel_Tiling_outProd<<<grid1, block1>>>(A, B, C);
}
void test_mulMat_128x256x8(Mat* A, Mat* B, Mat* C){
    CHECK(cudaDeviceSynchronize());
    dim3 grid((C->width + 255) / 256, (C->height + 127) / 128);
    sgemm_128x256x8_kernel<<<grid, 256>>>(
        A->elements, B->elements, C->elements, A->height, B->width, B->height,
        B->height * sizeof(float) * 32,
        A->height * sizeof(float) * 8);
}

void test_kernel(int kernel_num, Mat* A, Mat* B, Mat* C, dim3 block, dim3 grid){
    CHECK(cudaDeviceSynchronize());
    switch (kernel_num)
    {
    case 0: test_cublas(A, B, C);break;
    case 1: test_mulMat_naive(A, B, C, block, grid); break;
    case 2: test_mulMat_1x4(A, B, C, block, grid); break;
    case 3: test_mulMat_4x4(A, B, C, block, grid); break;
    case 4: test_mulMat_Tiling(A, B, C, block, grid); break;
    case 5: test_mulMat_Tiling_Coalesing(A, B, C, block, grid); break;
    case 6: test_mulMat_Tiling_noBankconflict(A, B, C, block, grid); break;
    case 7: test_mulMat_outerproduct(A, B, C, block, grid); break; 
    case 8: test_mulMat_128x256x8(A, B, C); break;
    default:
        break;
    }
}