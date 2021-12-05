/*
	Author: Matthew Lepis
	Date:	10/20/2021
	Desc:	This program computes a histogram using a CUDA device's dynamically allocated shared memory and shared memory atomics
	CUDA C/C++
*/
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <helper_functions.h>
#include <helper_cuda.h>

#include <cstdlib>
#include <iostream>
#include <cassert>

#define cout std::cout
#define endl std::endl
#define cin std::cin

const int MAX = 1024;
const int testrunNum = 300;

__global__ void histogramShared(int* input, int* bins, int numElements, int numBins, int DIV)
{
	// Allocating Shared Memory
	extern __shared__ int shared_bins[]; // This is private for each block of threads
		/*
		*	'extern' is used here to indicate that we are declaring the size outside of the perview of the kernel, specifically in the 3rd cuda parameters of the kernel call.
		*/

	// Calculate a global thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	/*	  {						 }	{		   }
	//				 ^					 ^
	// Decides which block we are in	 |
	//						Decides which thread we are in, inside said block
	*/

	// Initialize our shared memory
	if (threadIdx.x < numBins) {
		shared_bins[threadIdx.x] = 0;
	}
	__syncthreads(); // Syncing threads so nothing tries to fill the shared memory before it has been initialized.

	// Boundary Check; only indexing on valid memory
	if (tid < numElements)
	{
		int bin = input[tid] / DIV; // DIV is the mapping value to the bins.
		atomicAdd(&shared_bins[bin], 1);
	}
	// Waiting for threads to finish binning
	__syncthreads(); // Ensures every thread block has completed their atomic adds before writing back to main memory.

	// Writing back partial-result bins (shared memory) to the main bins (global memory)
	if (threadIdx.x < numBins) {
		atomicAdd(&bins[threadIdx.x], shared_bins[threadIdx.x]);
	}
}

int* verify_result(int* input, int numElements, int numBins, int DIV)
{
	int *temp = input;
	int *returnBins = new int[numBins];
	for (int i = 0; i < numBins; i++)
		returnBins[i] = 0;

	// For every element
	for (int i = 0; i < numElements; i++)
	{
		int bin = temp[i] / DIV;
		returnBins[bin]++;
	}
	return returnBins;
}

void init_hist(int* in_array, int numElem, int maxNum, int* in_bins, int numBins)
{
	// filling array with random numbers.
	for (int i = 0; i < numElem; i++)
		in_array[i] = rand() % maxNum; // Ensures that the random number generated will fit into the alloted bins.
}

void makeHistogram(int num_bins, int num_elements, int thread_count)
{

	// Set # of elements to bin (2^10 default)
	int numElements = num_elements;
	size_t bytes = numElements * sizeof(int);

	// Set # of bins
	int numBins = num_bins;
	size_t bytes_bins = numBins * sizeof(int);

	//  Creating and Allocating Host Variables
	int* input_H, * bins_H;
	cudaHostAlloc(&input_H, bytes, cudaHostAllocDefault);
	cudaHostAlloc(&bins_H, bytes_bins, cudaHostAllocDefault);

	// Creating and Allocating Device variables.
	int* input_D, * bins_D;
	cudaMalloc(&input_D, bytes);
	cudaMalloc(&bins_D, bytes_bins);

	// Initializing Input array and bins
	init_hist(input_H, numElements, MAX, bins_H, numBins);

	// Initialize the bins
	for (int i = 0; i < numBins; i++)
		bins_H[i] = 0;

	// Set mapping function, divisor; divisor is padded for non-even division of bins
	int DIV = (MAX + numBins - 1) / numBins;

	// CTA and Grid dimensions
	int THREADS = thread_count;								//----------------------------------------------------------------------------------------
	int BLOCKS = (numElements + THREADS - 1) / THREADS;		//----------------------------------------------------------------------------------------

	// Setting the size of dynam allocated shared memory
	size_t bytes_sharedMem = bytes_bins; //= numBins * sizeof(int)

	// Copying Host Memory to Device Memory
	cudaMemcpy(input_D, input_H, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(bins_D, bins_H, bytes_bins, cudaMemcpyHostToDevice);

	// Test Timer creation
	StopWatchInterface* hTimer = NULL;
	sdkCreateTimer(&hTimer);

	for (int iter = -1; iter < testrunNum; iter++)
	{
		//iter == -1 -- warmup iteration
		if (iter == -1)
		{
			// Warm up and Validation.
			histogramShared <<<BLOCKS, THREADS, bytes_sharedMem>>> (input_D, bins_D, numElements, numBins, DIV);
			cudaMemcpy(bins_H, bins_D, bytes_bins, cudaMemcpyDeviceToHost);

			int* tempCPU = verify_result(input_H, numElements, numBins, DIV);
			for (int i = 0; i < numBins; i++)
				if (tempCPU[i] != bins_H[i])
				{
					cout << "ERROR, GPU AND CPU RESULTS OUT OF SYNC." << endl;
					cout << "tempCPU: " << tempCPU[i] << "\tbins_H: " << bins_H[i] << endl;
				}
			delete tempCPU;

			cout << "\nSuccessful Run, GPU and CPU results are in unison.\n" << endl;
		}
		else if (iter == 0)
		{
			sdkResetTimer(&hTimer);
			sdkStartTimer(&hTimer);
			histogramShared <<<BLOCKS, THREADS, bytes_sharedMem>>> (input_D, bins_D, numElements, numBins, DIV);
		}
		else
			histogramShared <<<BLOCKS, THREADS, bytes_sharedMem>>> (input_D, bins_D, numElements, numBins, DIV);
	}
	cudaDeviceSynchronize();
	sdkStopTimer(&hTimer);

	double dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer) / (double)testrunNum;
	printf("    histogram, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes, Workgroup = %u\n",
		(1.0e-6 * (double)bytes / dAvgSecs), dAvgSecs, bytes, thread_count);


	int overflowThreads = numElements % THREADS;
	int AtomicOps = numElements + ((((BLOCKS - 1) * THREADS) + overflowThreads) * numBins);
	double GigaAtomicOpsPerSec = (double)AtomicOps / (dAvgSecs * 1000000000);

	cout << "    Giga Atomic Operations per second: " << GigaAtomicOpsPerSec << " GAOPS" << endl;

	printf("\n...reading back GPU results...\n\nHistogram Output:\n");

	int temp = 0;
	for (int i = 0; i < numBins; i++)
	{
		temp += bins_H[i];
		cout << bins_H[i] << " ";
	}

	sdkDeleteTimer(&hTimer);
	cudaFree(input_D);
	cudaFree(bins_D);
	cudaFreeHost(input_H);
	cudaFreeHost(bins_H);
}

int main(int argc, char** argv)
{
	int numBins;
	int numElements;
	int threadCount;

	if (argc == 4)
	{
		assert(atoi(argv[1]) > 1 || atoi(argv[1]) < 9);
		numBins = 2 << (atoi(argv[1]) - 1);

		assert(atoi(argv[2]) > 0);
		numElements = atoi(argv[2]);

		assert(atoi(argv[3]) > 0 || atoi(argv[3]) < 1025);
		threadCount = atoi(argv[3]);

		cout << "Command Line Arguments Accepted." << endl;
	}
	else
	{
		cout << "Enter the desired amount of bins, where that value is 'k' in: 2^k: " << endl;
		cout << "\tBetween 2 and 8, inclusive." << endl;
		cin >> numBins;
		assert(numBins > 1 || numBins < 9);
		numBins = 2 << (numBins - 1);

		cout << "Enter the desired number of elements in your input vector: " << endl;
		cout << "\tGreater than 0." << endl;
		cin >> numElements;
		assert(numElements > 0);

		cout << "Enter the desired thread count per block: " << endl;
		cout << "\tBetween 256 and 1024, inclusive." << endl;
		cin >> threadCount;
		assert(threadCount > 0 || threadCount < 1025);
	}

	makeHistogram(numBins, numElements, threadCount);

	return 0;
}