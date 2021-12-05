#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <cassert>
#include <iostream>


using namespace std;

#define valRANGE 100

__global__ void histogram(int* input, int* bins, int N, int N_bins, int DIV) // N is number of elements to bin
{
	// Calculate the global thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Boundary Check
	if (tid < N) // Making sure that the thread id is less than the number of elements that needs binning
	{
		int bin = input[tid] / DIV; // DIV is the mapping value to the bins.
		atomicAdd(&bins[bin], 1);
	}
}

// Initialize our input array w/ random number from 0-99
void initInputArray(int* a, int N)
{
	for (int i = 0; i < N; i++)
	{
		a[i] = rand() % valRANGE;
	}
}


int main()
{
	// Number of elements to bin(2^20 default)
	int numElements = 1 << 10;
	size_t bytes = numElements * sizeof(int);

	// Select the number of bins
	int N_bins = 10;
	size_t bytes_bins = N_bins * sizeof(int);

	// Allocate memory Input array and bins
	int *input_H, *bins_H;

	input_H = (int*)malloc(bytes);
	bins_H = (int*)malloc(bytes_bins);

	// Initialize our data into input array
	initInputArray(input_H, numElements);

	// Set the divisor for finding the correspoinding bin for an input
	int DIV = (valRANGE + N_bins - 1) / N_bins; // valRANGE is the number of possible values (0 through 99)


// set bins to 0, maybe change this to kernel
	for (int i = 0; i < N_bins; i++)
	{
		bins_H[i] = 0;
	}

	// Set the dimensions of our CTA and Grid
	int THREADS = 512;
	int BLOCKS = (numElements + THREADS - 1) / THREADS;

	// Allocating memory on device
	int* input_D, * bins_D;
	cudaMalloc(&input_D, bytes);
	cudaMalloc(&bins_D, bytes_bins);

	// Copying from Host to Device memory
	cudaMemcpy(input_D, input_H, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(bins_H, bins_D, bytes_bins, cudaMemcpyHostToDevice);


	histogram <<<BLOCKS, THREADS>>>(input_D, bins_D, numElements, N_bins, DIV);
	

	cudaMemcpy(input_H, input_D, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(bins_H, bins_D, bytes_bins, cudaMemcpyDeviceToHost);


	int temp = 0;
	for (int i = 0; i < N_bins; i++)
	{
		temp += bins_H[i];
	}
	cout << temp << endl;

	free(input_H);
	free(bins_H);
	cudaFree(input_D);
	cudaFree(bins_D);
	return 0;
}