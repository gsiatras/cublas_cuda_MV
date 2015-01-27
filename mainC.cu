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

__global__ void MultMVOptimizedKernel(double *c, double *b_input, double *A, const int M, const int N, const int BLOCK_WIDTH) {
	__shared__ int blockElt;  // holds the current block width
	__shared__ int blockxInd;
	__shared__ int blockyInd;

	// Run this only 1 time
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

	extern	__shared__ double b[];

	// Copy part of b into shared mem
	if (threadIdx.x < blockElt) {
		b[threadIdx.x] = b_input[blockxInd + threadIdx.x];
	}

	__syncthreads();

	double cSum = (double) 0;
	int threadyInd = blockyInd + threadIdx.x;

	// Access matrix verticallly
	if (threadyInd < M) {
		// For every element in the block row
		for (int i = 0; i < blockElt; i++){
			cSum += b[i] * A[(blockxInd + i) * (M) + (threadyInd)];
		}

		// Atomic add the temp sum to the c vector
		atomicAdd(c + threadyInd, cSum);
	}
}

int main(int argc, char ** argv) {
	int M, N;
	int BLOCK_WIDTH = 32;
	
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

	// Calculate "dynamic" BLOCK_WIDTH
	if (N <= 128)
		BLOCK_WIDTH = 32;
	else if (N <= 256)
		BLOCK_WIDTH = 64;
	else if (N <= 512)
		BLOCK_WIDTH = 128;
	else if (N <= 1024)
		BLOCK_WIDTH = 224;
	else if (N <= 2048)
		BLOCK_WIDTH = 416;
	else
		BLOCK_WIDTH = 512;

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
	MultMVOptimizedKernel<<<dimGrid, dimBlock, sharedMem>>>(d_c, d_b, d_A, M, N, BLOCK_WIDTH);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	fprintf(stdout, "Execution completed. Elapsed Time = %6.8f msecs\n", milliseconds);

	// Get results from the device
	cudaMemcpy(h_c, d_c, M * sizeof(double), cudaMemcpyDeviceToHost);

	// Free host memory
	free(h_A); free(h_b); free(h_c);
	// Free GPU memory
	cudaFree(d_A); cudaFree(d_b); cudaFree(d_c);

	return EXIT_SUCCESS;
}
