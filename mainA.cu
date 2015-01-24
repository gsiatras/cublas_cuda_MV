// System Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Cuda Includes
#include <cublas.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// auxiliary functions
#include "AuxFuncs.h"

// Macro to store elements in a linear space in row-major format
#define IDX2C(i, j, ld) (((i) * (ld)) + (j))


int main(int argc, char ** argv) {
	int M, N = 0;
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

	cublasStatus status;
	double * h_A, * h_b, * h_c; // host copies of a, b, c
	double * d_A, * d_b, * d_c; // device copies of a, b, c
	d_A = d_b = d_c = 0;

	const double alf = 1, bet = 0;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Initialize CUBLAS
	fprintf(stdout, "Using cublasDgemv() test running..\n");
	status = cublasInit();
	if (status != CUBLAS_STATUS_SUCCESS) {
		_error_handler("CUBLAS initialization error\n");
	}

	// Allocate host memory for the matrices
	((h_A = (double *) malloc(M * N * sizeof(double))) != 0) ?
	((h_b = (double *) malloc(N * sizeof(double))) != 0) ?
	((h_c = (double *) malloc(M * sizeof(double))) != 0) ?
	:
	_error_handler("host memory allocation error (C)\n") :
	_error_handler("host memory allocation error (B)\n") :
	_error_handler("host memory allocation error (A)\n") ;

	// Allocate device memory for the matrices
	((status = cublasAlloc(M * N, sizeof(double), (void**)&d_A)) == CUBLAS_STATUS_SUCCESS) ?
	((status = cublasAlloc(N, sizeof(double), (void**)&d_b)) == CUBLAS_STATUS_SUCCESS) ?
	((status = cublasAlloc(M, sizeof(double), (void**)&d_c)) == CUBLAS_STATUS_SUCCESS) ?
	:
	_error_handler("device memory allocation error (C)\n") :
	_error_handler("device memory allocation error (B)\n") :
	_error_handler("device memory allocation error (A)\n") ;
	
	// Initialize matrix A and vector b with some values and also zero-ize c vector
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			h_A[i*N + j] = randDouble();
		}
	}

	for (int i = 0; i < N; i++) {
		h_b[i] = randDouble();
	}

	for (int i = 0; i < M; i++) {
		h_c[i] = 0;
	}

	// Initialize the device matrices with the host matrices
	((status = cublasSetMatrix(M, N, sizeof(double), h_A, M, d_A, M)) == CUBLAS_STATUS_SUCCESS) ?
	((status = cublasSetVector(N, sizeof(double), h_b, 1, d_b, 1)) == CUBLAS_STATUS_SUCCESS) ?
	((status = cublasSetVector(M, sizeof(double), h_c, 1, d_c, 1)) == CUBLAS_STATUS_SUCCESS) ?
	:
	_error_handler("device access error (write C)\n") :
	_error_handler("device access error (write B)\n") :
	_error_handler("device access error (write A)\n") ;

	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);
	
	// Performs operation using CUBLAS
	cudaEventRecord(start);
	cublasDgemv(handle, CUBLAS_OP_T, N, M, &alf, d_A, N, d_b, 1, &bet, d_c, 1);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	fprintf(stdout, "Execution completed. Elapsed Time = %6.8f msecs\n", milliseconds);

	// Destroy the handle
	cublasDestroy(handle);

	if ((status = cublasGetError()) != CUBLAS_STATUS_SUCCESS) {
		_error_handler("kernel execution error.\n");
		return EXIT_FAILURE;
	}

	// Read the result back
	status = cublasGetVector(M, sizeof(h_c[0]), d_c, 1, h_c, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
		_error_handler("device access error (read C)\n");
	}

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

	// Shutdown
	status = cublasShutdown();
	if (status != CUBLAS_STATUS_SUCCESS) {
		_error_handler("shutdown error (A)\n");
	}

	return EXIT_SUCCESS;
}
