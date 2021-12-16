#include "GEMM.h"

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
        printf("Arrays not match!!!!.\n\n");
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
	unsigned int col = blockIdx.x * blockDim.x * 4  + threadIdx.x *4;
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
    unsigned int row = (blockIdx.y * blockDim.y + threadIdx.y)*4 ;	
	unsigned int col = (blockIdx.x * blockDim.x + threadIdx.x)*4;
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



__global__ void MatMulKernel_Tiling_outProd(Mat *A, Mat *B, Mat *C) {
	/* Basic tiling implementation of matrix multiplication.
	 * Based on a more mathematically reasonable indexing method.
	 */
	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float As[TILE_SIZE * TILE_SIZE];
    float cv[TILE_SIZE] = {0};

	int aBegin = A->width * TILE_SIZE * by;
	int aEnd = aBegin + A->width - 1;
	int aStep = TILE_SIZE;

	int bBegin = TILE_SIZE * VECTOR_SIZE * bx;
	int bStep = TILE_SIZE * B->width;


    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
            // Load Asub with size of TILE*TILE in colomn-major style.
            // Each thread needs to load TILE_SIZE / VECTOR_SIZE values of A.
            int t = VECTOR_SIZE;
            for (int i = 0; i < TILE_SIZE / VECTOR_SIZE; ++i) {
                As[ (i*t+ty) + TILE_SIZE * tx] = A->elements[a + A->width*(i*t+ty) + tx];
            }
            __syncthreads();

            float *ap = As;	// Point to the first address of As, increase later.
            // TODO: global memory ? register ? not clear :(
            float *bp = &B->elements[b + TILE_SIZE * ty + tx];	

            for (int i = 0; i < TILE_SIZE; ++i) {
                float bv = *bp;	
            // Each thread calculate a vector of C with size of TILE_SIZE.
                for (int j = 0; j < TILE_SIZE; ++j) {
                    cv[j] += ap[j] * bv;
                }
                ap += TILE_SIZE;
                bp += B->width;
            }
            __syncthreads();
        }
        
        // Store each value of Csub back to C in global memory.
        int c = B->width * TILE_SIZE * by + TILE_SIZE * VECTOR_SIZE * bx;
        c += TILE_SIZE * ty + tx;
        for (int i = 0; i < TILE_SIZE; ++i) {
            C->elements[c] = cv[i];
            c += B->width;
        }
}

__global__ void MatMulKernel_Tiling_Coalesing(Mat *A, Mat *B, Mat *C) {
	/* Basic tiling implementation of matrix multiplication.
	 * Based on a more mathematically reasonable indexing method.
	 */
	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float As[TILE_SIZE][TILE_SIZE];
	__shared__ float Bs[TILE_SIZE][TILE_SIZE];

	int aBegin = A->width * TILE_SIZE * by;
	int aEnd = aBegin + A->width - 1;
	int aStep = TILE_SIZE;

	int bBegin = TILE_SIZE * bx;
	int bStep = TILE_SIZE * B->width;

	float Csub = 0;

	for (int i = aBegin, j = bBegin; i <= aEnd; i += aStep, j += bStep) {
        // calculate tile C_ij
        // record tile A_ik and tile B00 B_kj
		As[ty][tx] = A->elements[i + A->width * ty + tx];
		Bs[tx][ty] = B->elements[j + B->width * ty + tx];

		__syncthreads();

		for (int k = 0; k < TILE_SIZE; ++k) {
			Csub += As[ty][k]*Bs[tx][k];
		}
		
		__syncthreads();
	}
	int cIdx = C->width * TILE_SIZE * by + TILE_SIZE * bx;
	C->elements[cIdx + C->width * ty + tx] = Csub;
}

__global__ void MatMulKernel_Tiling_noBankflict(Mat *A, Mat *B, Mat *C) {
	/* Basic tiling implementation of matrix multiplication.
	 * Based on a more mathematically reasonable indexing method.
	 */
	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float As[TILE_SIZE][TILE_SIZE];
	__shared__ float Bs[TILE_SIZE][TILE_SIZE];

	int aBegin = A->width * TILE_SIZE * by;
	int aEnd = aBegin + A->width - 1;
	int aStep = TILE_SIZE;

	int bBegin = TILE_SIZE * bx;
	int bStep = TILE_SIZE * B->width;

	float Csub = 0;

	for (int i = aBegin, j = bBegin; i <= aEnd; i += aStep, j += bStep) {
        // calculate tile C_ij
        // record tile A_ik and tile B00 B_kj
		As[ty][tx] = A->elements[i + A->width * ty + tx];
		Bs[ty][tx] = B->elements[j + B->width * ty + tx];

		__syncthreads();

		for (int k = 0; k < TILE_SIZE; ++k) {
			Csub += As[ty][k]*Bs[k][tx];
		}
		
		__syncthreads();
	}
	int cIdx = C->width * TILE_SIZE * by + TILE_SIZE * bx;
	C->elements[cIdx + C->width * ty + tx] = Csub;
}

__global__ void MatMulKernel_Tiling(Mat *A, Mat *B, Mat *C) {
	/* Basic tiling implementation of matrix multiplication.
	 * Based on a more mathematically reasonable indexing method.
	 */
	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float As[TILE_SIZE][TILE_SIZE];
	__shared__ float Bs[TILE_SIZE][TILE_SIZE];

	int aBegin = A->width * TILE_SIZE * by;
	int aEnd = aBegin + A->width - 1;
	int aStep = TILE_SIZE;

	int bBegin = TILE_SIZE * bx;
	int bStep = TILE_SIZE * B->width;

	float Csub = 0;

	for (int i = aBegin, j = bBegin; i <= aEnd; i += aStep, j += bStep) {
        // calculate tile C_ij
        // record tile A_ik and tile B00 B_kj
		As[ty][tx] = A->elements[i + A->width * ty + tx];
		Bs[tx][ty] = B->elements[j + B->width * tx + ty];

		__syncthreads();

		for (int k = 0; k < TILE_SIZE; ++k) {
			Csub += As[ty][k]*Bs[k][tx];
		}
		
		__syncthreads();
	}
	int cIdx = C->width * TILE_SIZE * by + TILE_SIZE * bx;
	C->elements[cIdx + C->width * ty + tx] = Csub;
}