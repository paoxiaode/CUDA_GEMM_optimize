#include "include/common.h"
#include "include/GEMM.h"

#include <cuda_runtime.h>
#include <stdio.h>



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

    const int nxk = nx * nk;
    const int nky = nk * ny;
    const int nxy = nx *ny;
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
    //////////////////////////////
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((ny + block.x - 1)/block.x, (nx + block.y - 1)/block.y);
    iStart = seconds();
    warmup<<<grid, block>>>(A, B, kernel1);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    //printf("mulMat on warmup: \t%f\n", iElaps);
    CHECK(cudaGetLastError());
    //////////////////////////////
    iStart = seconds();
    MatMulKernel<<<grid, block>>>(A, B, kernel2);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("mulMat on device orgin: \t%f\n", iElaps);
    CHECK(cudaGetLastError());
    checkResult(kernel2, kernel1);
    cleardata(kernel2->elements, nxy);
    //////////////////////////////
    iStart = seconds();
    dim3 block1(block.x/4, block.y);
    MatMulKernel_1x4<<<grid, block1>>>(A, B, kernel2);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("mulMat on device 1*4: \t%f\n", iElaps);
    CHECK(cudaGetLastError());
    checkResult(kernel2, kernel1);
    cleardata(kernel2->elements, nxy);

    ////////////////////////////
    iStart = seconds();
    dim3 block2(block.x/4, block.y/4);
    MatMulKernel_4x4<<<grid, block2>>>(A, B, kernel2);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("mulMat on device 4*4: \t%f\n", iElaps);
    CHECK(cudaGetLastError());
    checkResult(kernel2, kernel1);
    cleardata(kernel2->elements, nxy);

    //////////////////////////////
    iStart = seconds();
    MatMulKernel_Tiling<<<grid, block>>>(A, B, kernel2);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("mulMat on device tiling: \t%f\n", iElaps);
    CHECK(cudaGetLastError());
    checkResult(kernel2, kernel1);
    cleardata(kernel2->elements, nxy);

    //////////////////////////////
    iStart = seconds();
    MatMulKernel_Tiling_Coalesing<<<grid, block>>>(A, B, kernel2);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("mulMat on device tiling coalesing: \t%f\n", iElaps);
    CHECK(cudaGetLastError());
    checkResult(kernel2, kernel1);
    cleardata(kernel2->elements, nxy);

    //////////////////////////////
    iStart = seconds();
    MatMulKernel_Tiling_noBankflict<<<grid, block>>>(A, B, kernel2);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("mulMat on device tiling noBankflict: \t%f\n", iElaps);
    CHECK(cudaGetLastError());
    checkResult(kernel2, kernel1);
    cleardata(kernel2->elements, nxy);

    //////////////////////////////
    iStart = seconds();
    dim3 block3(TILE_SIZE, VECTOR_SIZE);
    dim3 grid1((ny + block.x - 1)/(block1.x*block1.y), (nx + block.y - 1)/TILE_SIZE);
    MatMulKernel_Tiling_outProd<<<grid1, block3>>>(A, B, kernel2);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("mulMat on device outter product: \t%f\n", iElaps);
    CHECK(cudaGetLastError());
    checkResult(kernel2, kernel1);
    return 0;
}