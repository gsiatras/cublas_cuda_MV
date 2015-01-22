// System Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Cuda Includes
#include <cublas.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// Macro to store elements in a linear space in row-major format
#define IDX2C(i, j, ld) (((i) * (ld)) + (j))

// TODO:
/*
* Produce large scale matrices with 
* controllable resulting vector c
*/

int cuberror_handler(const char * _error) {
	fprintf(stderr, "%s\n", _error);
	exit(EXIT_FAILURE);
}

// Print matrix/vector A(nr_rows_A, nr_cols_A) in row-major format
void print_data(const double * A, int nr_rows_A, int nr_cols_A) {

	for (int i = 0; i < nr_rows_A; i++) {
		for (int j = 0; j < nr_cols_A; j++) {

			fprintf(stdout, "%1.0f ", A[IDX2C(i, j, nr_cols_A)]);
		}
		//fprintf(stdout, "\n");
	}
	//fprintf(stdout, "\n");
}

// Main func
int main(int argc, char ** argv) {
	int M, N = 0;
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
	double * h_A, * h_B, * h_C;
	double * d_A, * d_B, * d_C;
	const double alf = 1, bet = 0;
	float deltaT = 0.0;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	d_A = d_B = d_C = 0;
	// Initialize CUBLAS
	fprintf(stdout, "Using cublasSgemv() test running..\n");
	status = cublasInit();
	if (status != CUBLAS_STATUS_SUCCESS) {
		cuberror_handler("CUBLAS initialization error\n");
	}

    // Allocate host memory for the matrices
    ((h_A = (double *) malloc(M * N * sizeof(h_A[0]))) != 0) ?
    ((h_B = (double *) malloc(N * sizeof(h_B[0]))) != 0) ?
    ((h_C = (double *) malloc(M * sizeof(h_C[0]))) != 0) ?
    :
    cuberror_handler("host memory allocation error (C)\n") :
    cuberror_handler("host memory allocation error (B)\n") :
    cuberror_handler("host memory allocation error (A)\n") ;

	// Initialize matrix A and vector b with some values
	// and also zero-ize c vector
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			h_A[i*N + j] = i;
			//fprintf(stdout, "%1.0f ", h_A[IDX2C(i, j, N)]);
		}
	}

	fprintf(stdout, "\n");
	for (int i = 0; i < N; i++) {
		h_B[i] = 1;//rand() % 4;
	}
	//fprintf(stdout, "\n\n\n\n");
	for (int i = 0; i < M; i++) {
		h_C[i] = 0;
	}

	//print_data(h_A, M, N);
	//print_data(h_B, N, 1);

	// Allocate device memory for the matrices
	((status = cublasAlloc(M * N, sizeof(d_A[0]), (void**)&d_A)) == CUBLAS_STATUS_SUCCESS) ?
	((status = cublasAlloc(N, sizeof(d_B[0]), (void**)&d_B)) == CUBLAS_STATUS_SUCCESS) ?
	((status = cublasAlloc(M, sizeof(d_C[0]), (void**)&d_C)) == CUBLAS_STATUS_SUCCESS) ?
	:
	cuberror_handler("device memory allocation error (C)\n") :
	cuberror_handler("device memory allocation error (B)\n") :
	cuberror_handler("device memory allocation error (A)\n") ;


	// Initialize the device matrices with the host matrices
	((status = cublasSetMatrix(M, N, sizeof(h_A[0]), h_A, M, d_A, M)) == CUBLAS_STATUS_SUCCESS) ?
	((status = cublasSetVector(N, sizeof(h_B[0]), h_B, 1, d_B, 1)) == CUBLAS_STATUS_SUCCESS) ?
	((status = cublasSetVector(M, sizeof(h_C[0]), h_C, 1, d_C, 1)) == CUBLAS_STATUS_SUCCESS) ?
	:
	cuberror_handler("device access error (write C)\n") :
	cuberror_handler("device access error (write B)\n") :
	cuberror_handler("device access error (write A)\n") ;

	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);
	
	/* Performs operation using CUBLAS */
	cudaEventRecord(start);
	cublasDgemv(handle, CUBLAS_OP_T, N, M, &alf, d_A, N, d_B, 1, &bet, d_C, 1);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	fprintf(stdout, "Done.. Elapsed Time = %6.8f msecs\n", milliseconds);
	// Destroy the handle
	cublasDestroy(handle);

	if ((status = cublasGetError()) != CUBLAS_STATUS_SUCCESS) {
		cuberror_handler("kernel execution error.\n");
		return EXIT_FAILURE;
	}

	// Read the result back
	status = cublasGetVector(M, sizeof(h_C[0]), d_C, 1, h_C, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
		cuberror_handler("device access error (read C)\n");
	}
	fprintf(stdout, "Final vector result: \n");
	//print_data(h_C, M, 1);
	fprintf(stdout, "\nvector c: ");
	for (int i = 0; i < M; i++) {
		fprintf(stdout, "%1.0f ", h_C[i]);
	}

	fprintf(stdout, "\n A: ");
	for (int i = 0; i < 2*N; i++) {
		fprintf(stdout, "%1.0f ", h_A[i]);
	}


	// Free host memory
	free(h_A); free(h_B); free(h_C);
	// Free GPU memory
	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

	// Shutdown
	status = cublasShutdown();
	if (status != CUBLAS_STATUS_SUCCESS) {
		cuberror_handler("shutdown error (A)\n");
	}

	fprintf(stdout, "\nPress ENTER to exit...\n");
	getchar();

	return EXIT_SUCCESS;
}

