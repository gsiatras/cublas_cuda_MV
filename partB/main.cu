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

// Macro to store elements in a linear space in row-major format
#define IDX2C(i, j, ld) (((i) * (ld)) + (j))

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
	}
}


__global__ void my_kernel(double *mat, double *vec, double *out, const int m, const int n){
    int row = threadIdx.x+blockIdx.x*blockDim.x;
    
    double sum = 0;
    if(row < m){
        for(int i=0; i < n; i++)
            sum += vec[i] * mat[ row * n + i] ;
        out[ row ] = sum;
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
		fprintf(stderr, "USAGE: main <matrixHeight> <matrixWidth>\n");
		exit(-1);
	}

	double * h_A, * h_B, * h_C;
	double * d_A, * d_B, * d_C;


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	d_A = d_B = d_C = 0;
	// Initialize CUBLAS

    // Allocate host memory for the matrices
    ((h_A = (double *) malloc(M * N * sizeof(double))) != 0) ?
    ((h_B = (double *) malloc(N * sizeof(double))) != 0) ?
    ((h_C = (double *) malloc(M * sizeof(double))) != 0) ?
    :
    cuberror_handler("!!!! host memory allocation error (C)\n") :
    cuberror_handler("!!!! host memory allocation error (B)\n") :
    cuberror_handler("!!!! host memory allocation error (A)\n") ;

	// Initialize matrix A and vector b with some values
	// and also zero-ize c vector
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			h_A[i*N + j] = i;
	 		//fprintf(stdout, "%f ", h_A[IDX2C(i, j, N)]);
		}
		//fprintf(stdout, "\n");
	}

	fprintf(stdout, "\n");
	for (int i = 0; i < N; i++) {
		h_B[i] = 1;
	}

	for (int i = 0; i < M; i++) {
		h_C[i] = 0;
	}

	//print_data(h_A, M, N);
	//print_data(h_B, N, 1);

    cudaMalloc((void **) &d_A, sizeof(double) * M * N);
    cudaMalloc((void **) &d_B, sizeof(double) * N);
    cudaMalloc((void **) &d_C, sizeof(double) * M);

    cudaMemcpy(d_A, h_A, sizeof(double) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(double) * N, cudaMemcpyHostToDevice);

	unsigned int blocksize = 512; // or any size up to 512
	unsigned int nblocks = M / blocksize + 1;	   

	float milliseconds = 0;

	cudaEventRecord(start);
	my_kernel<<<nblocks, blocksize>>>(d_A, d_B, d_C, M, N);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&milliseconds, start, stop);
	fprintf(stdout, "Done.. Elapsed Time = %6.8f msecs\n", milliseconds);


	cudaMemcpy(h_C, d_C, M * sizeof(h_C[0]), cudaMemcpyDeviceToHost);

	fprintf(stdout, "Final vector result: \n");
	//print_data(h_C, M, 1);
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


	fprintf(stdout, "\nPress ENTER to exit...\n");
	getchar();

	return EXIT_SUCCESS;
}

