// System Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Cuda Includes
#include <cublas.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "cuPrintf.cu"

// Macro to store elements in a linear space in row-major format
#define IDX2C(i, j, ld) (((i) * (ld)) + (j))

// #define N (2048*2048)
#define BLOCK_HEIGHT 512
#define BLOCK_WIDTH 64

__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +  __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

__global__ void MatMulKernel(double *out, double *in, double *a, const int matrixHeight, const int matrixWidth) {
  // get variables for loop
  // copy section of b into shared mem
  // go through the threads vertically and sum them into a variable
  // atomic add these variables to the corresponding c index

  // looping is happening horizontally on the matrix
  // BLOCK_WIDTH is again horizontal
  // BLOCK_HEIGHT is going vertical
  // n / BLOCK_WIDTH blocks horizontally
  // m / BLOCK_HEIGHT block vertically

  // get variables for loop
  // variable for loop length: blockEltHeight
  __shared__ int blockElt;
  __shared__ int blockxInd;
  __shared__ int blockyInd;
  if (threadIdx.x == 0) {
    if ((blockIdx.x + 1) * BLOCK_WIDTH <= matrixWidth)
      blockElt = BLOCK_WIDTH;
    else blockElt = matrixWidth % BLOCK_WIDTH;
    blockxInd = blockIdx.x * BLOCK_WIDTH;
    blockyInd = blockIdx.y * BLOCK_HEIGHT;
  }
  
  __syncthreads();
  
  // copy section of b into shared mem
  // use the first BLOCK_WIDTH of thread
  __shared__ double b[BLOCK_WIDTH];

  if (threadIdx.x < blockElt) 
    b[threadIdx.x] = in[blockxInd + threadIdx.x];
  
  __syncthreads();

  // summing variable
  double cSum = (double) 0;
  int threadyInd = blockyInd + threadIdx.x;

  // make sure we are inside the matrix verticallly
  if (threadyInd < matrixHeight) {
  
    // go through the threads vertically and sum them into a variable
    for (int i=0; i<blockElt; i++)
      // A col index   : blockIdx.x * BLOCK_WIDTH + i : blockxInd + i
      // A row index  : blockIdx.y * BLOCK_HEIGHT + threadIdx.x : blockyInd + threadIdx.x : threadyInd
      // B index : b[i]

      // cSum = B index * ( A col index * matrixHeight + A row index)
      cSum += b[i] * a[(blockxInd + i) * (matrixHeight) + (threadyInd)];
      //printf("csum = %f\n", cSum);
    
    // atomic add these variables to the corresponding c index
  	//cuPrintf("%d %1.0f\n",threadyInd, cSum);
    atomicAdd(out + threadyInd, cSum);
  }
  
}

int main(int argc, char ** argv) {
  int M, N;
  if (argc == 3){
    M = atoi(argv[1]);
    N = atoi(argv[2]);
  }
  else {
    fprintf(stderr, "Insufficient command line arguments!\n");
    fprintf(stderr, "USAGE: main <M> <N>\n");
    exit(-1);
  }

	double * h_A, * h_b, * h_c; 		  // host copies of a, b, c
	double * dev_a, * dev_b, * dev_c; // device copies of a, b, c

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// allocate device copies of a, b, c
	cudaMalloc((void **) &dev_a, M * N * sizeof(double));
	cudaMalloc((void **) &dev_b, N * sizeof(double));
	cudaMalloc((void **) &dev_c, M * sizeof(double));

	h_A = (double *) malloc(M * N * sizeof(double));
	h_b = (double *) malloc(N * sizeof(double));
	h_c = (double *) malloc(M * sizeof(double));

	// Initialize matrix A and vector b with some values
	// and also zero-ize c vector

	// #define IDX2C(i, j, ld) (((i) * (ld)) + (j))
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < M; i++) {
			h_A[j*M + i] = i;
			//h_A[IDX2C(j, i, M)] = i;//rand() % 5;
	 		//fprintf(stdout, "%f ", h_A[IDX2C(i, j, N)]);
		}
		//fprintf(stdout, "\n");
	}

	fprintf(stdout, "\n");
	for (int i = 0; i < N; i++) {
		h_b[i] = 1;//rand() % 4;
	}

	fprintf(stdout, "\n");
	for (int i = 0; i < M; i++) {
		h_c[i] = 0;
	}

	cudaMemcpy(dev_a, h_A, M * N * sizeof(double), cudaMemcpyHostToDevice);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      fprintf(stderr, "Error: %s \n", cudaGetErrorString(err));
  }

	cudaMemcpy(dev_b, h_b, N * sizeof(double), cudaMemcpyHostToDevice);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
      fprintf(stderr, "Error: %s \n", cudaGetErrorString(err));
  }

	cudaMemcpy(dev_c, h_c, M * sizeof(double), cudaMemcpyHostToDevice);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
      fprintf(stderr, "Error: %s \n", cudaGetErrorString(err));
  }

	cudaPrintfInit ();


  dim3 threadsPerBlock(N/BLOCK_WIDTH + 1, M/BLOCK_HEIGHT + 1);
  int blockCols = (int) ceil(N / (double) BLOCK_WIDTH);
  int blockRows = (int) ceil(M / (double) BLOCK_HEIGHT);
  dim3 dimBlock(BLOCK_HEIGHT);
  dim3 dimGrid(blockCols, blockRows);
  int sharedMem = 3 * sizeof (int) + BLOCK_WIDTH * sizeof (float);

	cudaEventRecord(start);
	// M h N? kai gt +1? http://prntscr.com/4zqiyd

  MatMulKernel<<<dimGrid, dimBlock, sharedMem>>>(dev_c, dev_b, dev_a, M, N);
	//MVKernel_shm1_grammes <<< 80, threadsPerBlock >>> (dev_a, dev_b, dev_c);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	fprintf(stdout, "Done.. Elapsed Time = %6.8f msecs\n", milliseconds);

	cudaPrintfDisplay (stdout, true);
	cudaPrintfEnd ();

	cudaMemcpy(h_c, dev_c, M * sizeof(double), cudaMemcpyDeviceToHost);

	/*fprintf(stdout, "\nvector b: ");
	  for (double i = 0; i < M; i++) {
	  fprintf(stdout, "%d ", h_b[i]);
	  }*/
  
	fprintf(stdout, "\nvector c: ");
	for (int i = 0; i < M; i++) {
		fprintf(stdout, "%1.0f ", h_c[i]);
	}

		fprintf(stdout, "\n A: ");
	for (int i = 0; i < 2*N; i++) {
		fprintf(stdout, "%1.0f ", h_A[i]);
	}
	printf("\n\n");
	// fprintf(stdout, "\ndev_c: %1.0f \n\n", dev_c);
	//fprintf(stdout, "\n** Results ** %d %d %d %d \n", h_c[0], h_c[1], h_c[2], h_c[3]);

	free(h_A); free(h_b); free(h_c);
	cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);

	return 0;
}