/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Includes, cuda */
#include <cublas.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>

/* Matrix size - Dimensions => M x N */
#define M  (4)
#define N  (3)
#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

// TODO:
/*
* Na to kanoume na douleuei gia
* otidipote N, M diladi na min
* einai sqaure to A
*/

int cuberror_handler(const char * _error) {
	fprintf(stderr, "%s\n", _error);
	exit(EXIT_FAILURE);
}

// Print matrix/vector A(nr_rows_A, nr_cols_A) storage in column-major format
void print_data(const double * A, int nr_rows_A, int nr_cols_A) {

	for (int j = 0; j < nr_cols_A; j++) {
		for (int i = 0; i < nr_rows_A; i++) {
			/* NA VALOUME RESOLUTION STIN PRINT */
			fprintf(stdout, "%1.0f ", A[IDX2C(i, j, nr_rows_A)]);
		}
		fprintf(stdout, "\n");
	}
	fprintf(stdout, "\n");
}


/* Main */
int main(int argc, char ** argv) {

	cublasStatus status;
	double * h_A, * h_B, * h_C;
	double * d_A, * d_B, * d_C;
	const double alf = 1, bet = 0;
	float deltaT = 0.0;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	d_A = d_B = d_C = 0;
	// srand(123);
	/* Initialize CUBLAS */
	fprintf(stdout, "Using cublasSgemv() test running..\n");

	status = cublasInit();
	if (status != CUBLAS_STATUS_SUCCESS) {
		cuberror_handler("!!!! CUBLAS initialization error\n");
	}

	/* Allocate host memory for the matrices */
	h_A = (double *) malloc(M * N * sizeof(h_A[0]));
	if (h_A == 0) {
		cuberror_handler("!!!! host memory allocation error (A)\n");
	}
	h_B = (double *) malloc(N * sizeof(h_B[0]));
	if (h_B == 0) {
		cuberror_handler("!!!! host memory allocation error (B)\n");
	}
	h_C = (double *) malloc(M * sizeof(h_C[0]));
	if (h_C == 0) {
		cuberror_handler("!!!! host memory allocation error (C)\n");
	}

	/* Fill the matrices with test data MODULO 4 */
	/*
	for (int i = 0; i < N * M; i++) {
		h_A[i] = rand() % 4; // / (float)RAND_MAX;
		fprintf(stdout, "%1.0f ", h_A[i]);
	}
	*/
	for (int j = 0; j < N; j++) { 
		for (int i = 0; i < M; i++) { 
			h_A[IDX2C(i, j, M)] = rand() % 4;// (float)(i * M + j + 1); 
			fprintf(stdout, "%1.0f ", h_A[i]);
		} 
	}

	printf("\n");
	for (int i = 0; i < N; i++) {
		h_B[i] = rand() % 4;
		fprintf(stdout, "%1.0f ", h_B[i]);
	}
	printf("\n\n\n\n");
	for (int i = 0; i < M; i++) {
		h_C[i] = 0;
	}

	print_data(h_A, M, N);
	print_data(h_B, N, 1);
	//print_data(h_C, M, 1);

	/* Allocate device memory for the matrices */
	status = cublasAlloc(M * N, sizeof(d_A[0]), (void**) &d_A);
	if (status != CUBLAS_STATUS_SUCCESS) {
		cuberror_handler("!!!! device memory allocation error (A)\n");
	}
	status = cublasAlloc(N, sizeof(d_B[0]), (void**) &d_B);
	if (status != CUBLAS_STATUS_SUCCESS) {
		cuberror_handler("!!!! device memory allocation error (B)\n");
	}
	status = cublasAlloc(M, sizeof(d_C[0]), (void**) &d_C);
	if (status != CUBLAS_STATUS_SUCCESS) {
		cuberror_handler("!!!! device memory allocation error (C)\n");
	}
	/* Initialize the device matrices with the host matrices */
	status = cublasSetMatrix(M, N, sizeof(h_A[0]), h_A, M, d_A, M);
	if (status != CUBLAS_STATUS_SUCCESS) {
		cuberror_handler("!!!! device access error (write A)\n");
	}
	status = cublasSetVector(N, sizeof(h_B[0]), h_B, 1, d_B, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
		cuberror_handler("!!!! device access error (write B)\n");
	}
	status = cublasSetVector(M, sizeof(h_C[0]), h_C, 1, d_C, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
		cuberror_handler("!!!! device access error (write C)\n");
	}

	/* Clear last error */
	// cublasGetError();

	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);
	/* Performs operation using CUBLAS */
	cudaEventRecord(start);
	cublasDgemv(handle, CUBLAS_OP_N, M, N, &alf, d_A, M, d_B, 1, &bet, d_C, 1);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	fprintf(stdout, "Done.. Elapsed Time = %6.8f msecs\n", milliseconds);
	// Destroy the handle
	cublasDestroy(handle);

	status = cublasGetError();
	if (status != CUBLAS_STATUS_SUCCESS) {
		cuberror_handler("!!!! kernel execution error.\n");
	}

	/* Allocate host memory for reading back the result from device memory 
	h_C = (double *) malloc(M * sizeof(h_C[0]));
	if (h_C == 0) {
		cuberror_handler("!!!! host memory allocation error (C)\n");
	}
	*/
	/* Read the result back */
	status = cublasGetVector(M, sizeof(h_C[0]), d_C, 1, h_C, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
		cuberror_handler("!!!! device access error (read C)\n");
	}
	// printf("%1.0f %1.0f %1.0f <---", h_C[0], h_C[1], h_C[2]);
	std::cout << "Final vector result: \n";
	print_data(h_C, M, 1);

	/* Memory clean up */
	free(h_A); free(h_B); free(h_C);
	//Free GPU memory
	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

	/* Shutdown */
	status = cublasShutdown();
	if (status != CUBLAS_STATUS_SUCCESS) {
		cuberror_handler("!!!! shutdown error (A)\n");
	}

	fprintf(stdout, "\nPress ENTER to exit...\n");
	getchar();

	return EXIT_SUCCESS;
}
