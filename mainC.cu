// System Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Cuda Includes
#include <cuda.h>
#include <cublas.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// auxiliary functions
#include "AuxFuncs.h"

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

__global__ void MultMVOptimizedKernel(double *c, double *b_input, double *A, const int M, const int N) {
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
		if ((blockIdx.x + 1) * BLOCK_WIDTH <= N){
			blockElt = BLOCK_WIDTH;
		}
		else {
			blockElt = N % BLOCK_WIDTH;
		}
		blockxInd = blockIdx.x * BLOCK_WIDTH;
		blockyInd = blockIdx.y * BLOCK_HEIGHT;
	}

	__syncthreads();

	// copy section of b into shared mem
	// use the first BLOCK_WIDTH of thread
	__shared__ double b[BLOCK_WIDTH];

	if (threadIdx.x < blockElt) {
		b[threadIdx.x] = b_input[blockxInd + threadIdx.x];
	}

	__syncthreads();

	// summing variable
	double cSum = (double) 0;
	int threadyInd = blockyInd + threadIdx.x;

	// make sure we are inside the matrix verticallly
	if (threadyInd < M) {
		// go through the threads vertically and sum them into a variable
		for (int i = 0; i < blockElt; i++)
		// A col index   : blockIdx.x * BLOCK_WIDTH + i : blockxInd + i
		// A row index  : blockIdx.y * BLOCK_HEIGHT + threadIdx.x : blockyInd + threadIdx.x : threadyInd
		// B index : b[i]

		// cSum = B index * ( A col index * M + A row index)
		cSum += b[i] * A[(blockxInd + i) * (M) + (threadyInd)];
		//printf("csum = %f\n", cSum);

		// atomic add these variables to the corresponding c index
		atomicAdd(c + threadyInd, cSum);
	}
}

int main(int argc, char ** argv) {
	int M, N;
	// init the seed with current local time
	srand(time(NULL));

	// Get M - N values from arguments
	if (argc == 3){
		M = atoi(argv[1]);
		N = atoi(argv[2]);
	}
	else {
		fprintf(stderr, "Insufficient command line arguments!\n");
		fprintf(stderr, "USAGE: main <matrixHeight> <matrixWidth>\n");
		exit(-1);
	}

	double * h_A, * h_b, * h_c; // host copies of a, b, c
	double * d_A, * d_b, * d_c; // device copies of a, b, c

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Allocate host memory for the matrix and the vectors
	((h_A = (double *) malloc(M * N * sizeof(double))) != 0) ?
	((h_b = (double *) malloc(N * sizeof(double))) != 0) ?
	((h_c = (double *) malloc(M * sizeof(double))) != 0) ?
	:
	_error_handler("host memory allocation error (C)\n") :
	_error_handler("host memory allocation error (B)\n") :
	_error_handler("host memory allocation error (A)\n") ;

	// Allocate device memory for the matrix and the vectors
	cudaMalloc((void **) &d_A, M * N * sizeof(double));
	cudaMalloc((void **) &d_b, N * sizeof(double));
	cudaMalloc((void **) &d_c, M * sizeof(double));

	// Initialize matrix A and vector b with some values and also zero-ize c vector
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < M; i++) {
			h_A[j*M + i] = randDouble();
		}
	}

	for (int i = 0; i < N; i++) {
		h_b[i] = randDouble();
	}

	for (int i = 0; i < M; i++) {
		h_c[i] = 0;
	}

	// Copy data from host to device
	cudaMemcpy(d_A, h_A, M * N * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, N * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, h_c, M * sizeof(double), cudaMemcpyHostToDevice);

	int blockCols = (int) ceil(N / (double)BLOCK_WIDTH);
	int blockRows = (int) ceil(M / (double)BLOCK_HEIGHT);
	int sharedMem = 3 * sizeof(int) + BLOCK_WIDTH * sizeof(double);
	
	dim3 dimBlock(BLOCK_HEIGHT);
	dim3 dimGrid(blockCols, blockRows);

	// Run kernel and measure the time needed
	cudaEventRecord(start);
	MultMVOptimizedKernel<<<dimGrid, dimBlock, sharedMem>>>(d_c, d_b, d_A, M, N);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	fprintf(stdout, "Execution completed. Elapsed Time = %6.8f msecs\n", milliseconds);

	// Get results from the device
	cudaMemcpy(h_c, d_c, M * sizeof(double), cudaMemcpyDeviceToHost);

	fprintf(stdout, "Result: \n");
	for (int i = 0; i < M; i++) {
		fprintf(stdout, "%6.8f ", h_c[i]);
	}
	fprintf(stdout, "\n");

	/*fprintf(stdout, "\n A: ");
	for (int i = 0; i < 2*N; i++) {
		fprintf(stdout, "%1.0f ", h_A[i]);
	}*/

	// Free host memory
	free(h_A); free(h_b); free(h_c);
	// Free GPU memory
	cudaFree(d_A); cudaFree(d_b); cudaFree(d_c);

	return EXIT_SUCCESS;
}