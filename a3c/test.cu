#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>

using namespace std;

#define CHECK(call)					\
{							\
  const cudaError_t error = call;			\
  if (error != cudaSuccess)				\
  {							\
    printf("Error: %s:%d ", __FILE__, __LINE__);	\
    printf("code:%d, reason:%s\n", error,		\
	   cudaGetErrorString(error));			\
    exit(1);						\
  }							\
}

__global__ void kernel(float *A, int *B, int *T)
{
	const int tid = blockIdx.x*blockDim.x + threadIdx.x;

	const int ntid = T[tid];
	//printf("%d\n", T[tid]);
	//if (B[ntid] != tid) printf("%d, %d\n", B[ntid], T[tid]);
	
	A[B[ntid]] *= 5.0;
}


__global__ void p_kernel(float *A, int *B, int *T)
{
	const int tid = blockIdx.x*blockDim.x + threadIdx.x;
	const int ntid = T[tid];
	//printf("%d\n", T[tid]);
	
	A[B[ntid]] *= 5.0;
	
	printf("%d,%lld\n", tid, &A[B[ntid]]);
}

#define WARP 32
#define THREAD 1024
int main(int argc, char **argv)
{
	CHECK(cudaSetDevice(0));

	int mode = atoi(argv[1]);

	int *h_t = (int *)malloc(THREAD * sizeof(int));
	ifstream ifs(argv[2]);
	string str;
	// skip header line
	//getline(ifs, str);

	for (int i = 0; i < THREAD; ++i) {
		getline(ifs, str);
		h_t[i] = atoi(str.c_str());
		//cout << h_t[i] << endl;
	}

	int nElm = THREAD;
	size_t nByte = nElm*sizeof(float);
	
	float *h_A;
	int *h_B;
	h_A = (float *)malloc(nByte);
	h_B = (int *)malloc(nElm * sizeof(int));

	//for(int i = 0; i < THREAD; ++i)
	//		h_B[i] = i;

	//srand(1234);
	//for(int i = 0; i < 512; ++i) {
	//		int j = rand()%512;
	//	int t = h_B[i];
	//	h_B[i] = h_B[j];
	//	h_B[j] = t;
	//}
	for (int i = 0; i < 32; ++i) {
          for (int j = 0; j < 16; ++j) {
	    h_B[i*32+j] = i + j*64;
	    h_B[i*32+j+16] = i+32 + j*64;
	  }
	}
	 
	float *d_A;
	int *d_B;
	int *d_t;
	CHECK(cudaMalloc((float **)&d_A, nByte));
	CHECK(cudaMalloc((int **)&d_B, nByte));
	CHECK(cudaMalloc((int **)&d_t, nByte));

	CHECK(cudaMemcpy(d_A, h_A, nByte, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_B, h_B, nByte, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_t, h_t, nByte, cudaMemcpyHostToDevice));

	int iLen = THREAD;
	dim3 block(iLen);
	dim3 grid( (nElm + block.x - 1) / block.x);

	//for (int i = 0; i < 10000; ++i)

	if (mode == 0)
		kernel<<<grid, block>>>(d_A, d_B, d_t);
	else if (mode == 1)
		p_kernel<<<grid, block>>>(d_A, d_B, d_t);

	//CHECK(cudaGetLastError());

	CHECK(cudaMemcpy(h_A, d_A, nByte, cudaMemcpyDeviceToHost));
	
	CHECK(cudaFree(d_A));
	CHECK(cudaFree(d_B));
	CHECK(cudaFree(d_t));

	free(h_A);
	free(h_B);
	free(h_t);

	//cudaDeviceReset();
}
