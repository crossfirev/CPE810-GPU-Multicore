/*
	Author: Matthew Lepis

*/
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <cassert>
#include <iostream>

using std::cout;
using std::cin;
using std::endl;

constexpr auto TILE_WIDTH = 32;

// Implimentation inspired from: mzn.rft on StackOverflow: https://stackoverflow.com/questions/13896560/multiply-rectangular-matrices-in-cuda
__global__ void matrixMultiplyShared(float* a, float* b, float* c, dim3 dimsA, dim3 dimsB, dim3 dimsC)
{
	// Shared memory allocation
	__shared__ float sharedA[TILE_WIDTH][TILE_WIDTH];
	__shared__ float sharedB[TILE_WIDTH][TILE_WIDTH];
	
	// Utilizing simplier variable names
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Row and Col indexing
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;
	
	// Temporary C value for accumulation
	float Cvalue = 0.0;

	if (dimsA.x != dimsB.y) return;
	for (int i = 0; i < (int)(ceil((float)dimsA.x / TILE_WIDTH)); i++)
	{

		if (i * TILE_WIDTH + tx < dimsA.x && Row < dimsA.y)
			sharedA[ty][tx] = a[Row * dimsA.x + i * TILE_WIDTH + tx];
		else
			sharedA[ty][tx] = 0.0;


		if (i * TILE_WIDTH + ty < dimsB.y && Col < dimsB.x)
			sharedB[ty][tx] = b[(i * TILE_WIDTH + ty) * dimsB.x + Col];
		else
			sharedB[ty][tx] = 0.0;

		__syncthreads();

		if (Row < dimsA.y && Col < dimsB.x)
			for (int j = 0; j < TILE_WIDTH; j++)
				Cvalue += sharedA[ty][j] * sharedB[j][tx];

		__syncthreads();
	}

	if (Row < dimsC.y && Col < dimsC.x)
		c[Row * dimsC.x + Col] = Cvalue;
}

// Verify the result on the CPU
// Implementation inspired from: Costantino Grana on StackOverflow: https://stackoverflow.com/questions/47023651/multiplying-matrices-in-one-dimensional-arrays/47024269
void verify_result(float *a, float *b, float *c, 
	int numARows, int numAColumns, 
	int numBRows, int numBColumns, 
	int numCRows, int numCColumns)
{
	float tmp;

	// For every Row
	for (int i = 0; i < numARows; i++)
	{
		// For every Column
		for (int j = 0; j < numBColumns; j++)
		{
			// For every element in the row-col pair
			tmp = 0;
			for (int k = 0; k < numAColumns; k++)
				tmp += a[i * numAColumns + k] * b[k * numBColumns + j];

			assert(tmp == c[i * numBColumns + j]);
		}
	}

}

void initializeMatrix(float *m, int n)
{
	for (int i = 0; i < n; i++)
		m[i] = rand() % 100; // 0 to 99, inclusive
}

void printMatrix(float* m, int n)
{
	for (int i = 0; i < n; i++)
		cout << m[i] << " ";
}

void multiplyMatrix(const dim3& dimsA, const dim3& dimsB, const dim3& dimsC)
{
	// Setting up performance metrics.
	cudaStream_t stream;
	cudaEvent_t start, stop;


	// Getting bytes size requirement for memory allocation.
	size_t a_Bytes = (dimsA.y * dimsA.x) * sizeof(float);
	size_t b_Bytes = (dimsB.y * dimsB.x) * sizeof(float);
	size_t c_Bytes = (dimsC.y * dimsC.x) * sizeof(float);

	// Allocating HOST memory for the matrices.
	float* a_H, * b_H, * c_H;
	a_H = (float*)malloc(a_Bytes);
	b_H = (float*)malloc(b_Bytes);
	c_H = (float*)malloc(c_Bytes);

	// Initialize the matrixes.
	initializeMatrix(a_H, dimsA.y * dimsA.x);
	initializeMatrix(b_H, dimsB.y * dimsB.x);

	// Allocating DEVICE memory for the matrices.
	float* a_D, * b_D, * c_D;
	cudaMalloc(&a_D, a_Bytes);
	cudaMalloc(&b_D, b_Bytes);
	cudaMalloc(&c_D, c_Bytes);

	// Creating Performance Metric.
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

	// Copying matrices from HOST memory to DEVICE memory.
	cudaMemcpy(a_D, a_H, a_Bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(b_D, b_H, b_Bytes, cudaMemcpyHostToDevice);

	// Establishing grid and block dimensions.
	dim3 threads(TILE_WIDTH, TILE_WIDTH); // Threads per Block in both dimensions.

	// Padding blocks of both dimensions to catch non-divisible elements sizes.
	int blocks_x = (dimsB.x + threads.x - 1) / threads.x;
	int blocks_y = (dimsA.y + threads.y - 1) / threads.y;
	dim3 grid(blocks_x, blocks_y);

	// Creating and starting Timer
	cout << "Computing result using CUDA Kernel..." << endl;

	// Performing Warm up operation.
	matrixMultiplyShared <<<grid, threads, 0, stream>>> (a_D, b_D, c_D, dimsA, dimsB, dimsC);
	cudaStreamSynchronize(stream);

	// Record the start event
	cudaEventRecord(start, stream);

	// KERNEL CALL!
	int nIter = 300;

	for (int j = 0; j < nIter; j++)
		matrixMultiplyShared <<<grid, threads, 0, stream>>> (a_D, b_D, c_D, dimsA, dimsB, dimsC);

	// Record the stop event
	cudaEventRecord(stop, stream);
	// Wait for the stop event to complete
	cudaEventSynchronize(stop);

	float msecTotal = 0.0f;
	cudaEventElapsedTime(&msecTotal, start, stop);

	// Compute and print the performance
	float msecPerMatrixMul = msecTotal / nIter;
	double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) * static_cast<double>(dimsA.y) * static_cast<double>(dimsB.x);
	double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
	printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n", gigaFlops, msecPerMatrixMul, flopsPerMatrixMul, threads.x * threads.y);

	cudaDeviceSynchronize();
	cudaMemcpy(c_H, c_D, c_Bytes, cudaMemcpyDeviceToHost);

	//verify_result(a_H, b_H, c_H, dimsA.y, dimsA.x, dimsB.y, dimsB.x, dimsC.y, dimsC.x);
	//printMatrix(c_H, dimsC.x * dimsC.y);
	cout << "SUCCESSFUL EXECUTION." << endl;

	free(a_H);
	free(b_H);
	free(c_H);
	cudaFree(a_D);
	cudaFree(b_D);
	cudaFree(c_D);
}

int main(int argc, char** argv)
{
	/*  //	Setting the matrix dimensions.	//
		
		You can only multiply two matrices if their dimensions are compatible.

			Which means the NUM OF COLUMNS in matrix A must be the same as 
							NUM OF ROWS in matrix B.
				
			./TiledMatrixMul -i <rowDimA>  <colDimA>  <colDimB>
									m	       n		  k
					 
					 y   x   y   x     y   x
					(m × n)·(n × k) = (m × k)
					   A       B         C

					   rowDimB = colDimA
					   rowDimC = rowDimA
					   colDimC = colDimB

					   b_Row = a_Col
					   c_Row = a_Row
					   c_Col = b_Col
	*/
	dim3 dimsA;
	dim3 dimsB;
	dim3 dimsC;

	// Reading in the matrices dimensions
	if (argc == 4)
	{
		assert(atoi(argv[1]) <= 0);
		dimsA.y = atoi(argv[1]);

		assert(atoi(argv[2]) <= 0);
		dimsA.x = atoi(argv[2]);

		assert(atoi(argv[3]) <= 0);
		dimsB.x = atoi(argv[3]);

		cout << "Command Line Arguments Accepted." << endl;
	}
	else
	{
		cout << "Enter row dimensions for matrix A: " << endl;
		cin >> dimsA.y;

		cout << "Enter column dimensions for matrix A: " << endl;
		cin >> dimsA.x;

		cout << "Enter column dimensions for matrix B: " << endl;
		cin >> dimsB.x;
	}

	// Filling in the rest.
	dimsB.y = dimsA.x;
	dimsC.y = dimsA.y;
	dimsC.x = dimsB.x;

	multiplyMatrix(dimsA, dimsB, dimsC);

	return 0;
}
