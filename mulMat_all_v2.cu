#include "include/common.h"
#include "include/kernel_ampere.cuh"

#include "include/GEMM.cuh"
#include "include/utils.cuh"

#include <cuda_runtime.h>
#include <stdio.h>
#include <cublas_v2.h>


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
    int N = 10, n_count;
    
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
    float elapsed_time;
    //////////////////////////////
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((ny + block.x - 1)/block.x, (nx + block.y - 1)/block.y);
    warmup<<<grid, block>>>(A, B, kernel1);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    ////////////////////////////
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    for(int kernel_count = 0; kernel_count < KERNEL_NUM; ++ kernel_count)
    {
        test_kernel(kernel_count, A, B, kernel2, block, grid);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
        if(kernel_count != 0 && (!checkResult(kernel1, kernel2, 0))){
            printf("Failed to pass the correctness verification. Exited.\n");
            exit(-3);
        }
        cudaEventRecord(beg);
        for(n_count = 0; n_count < N; ++n_count){
            test_kernel(kernel_count, A, B, kernel2, block, grid);
        }
        cudaEventRecord(end);
        cudaEventSynchronize(beg);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, beg, end);
        elapsed_time /= 1000.;
        printf("Average elasped time: %f second, performance: %f GFLOPS.\n", elapsed_time/N,2.*1e-9*N*nx*ny*nk/elapsed_time);
        fflush(stdout);
        cleardata(kernel2);
        CHECK(cudaDeviceSynchronize());
    }

    return 0;
}