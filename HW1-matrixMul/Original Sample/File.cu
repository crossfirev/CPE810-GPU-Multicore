// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>
using namespace std;

int main()
{
	// Array size of 2^16 (65536 elements)
	int N = 256;
	size_t bytes = sizeof(int) * N;

	// Vectors for holding the host-side (CPU-side) data
	vector<int> a;		a.reserve(N);
	vector<int> b;		b.reserve(N);
	vector<int> c;		c.reserve(N);

	// Initialize random numbers in each array
	for (int i = 0; i < N; i++)
	{
		a.push_back(rand() % 100);
		b.push_back(rand() % 100);
	}

	// Allocate memory on the device
	int* d_a, * d_b, * d_c;
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	// Copy data from the host to the device (CPU -> GPU)
	cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

	// Threads per CTA (1024)
	int NUM_THREADS = 1 << 10;

	// CTAs per Grid
	// We need to launch at LEAST as many threads as we have elements
	// This equation pads an extra CTA to the grid if N cannot evenly be divided
	// by NUM_THREADS (e.g. N = 1025, NUM_THREADS = 1024)
	int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

	// Launch the kernel on the GPU
	// Kernel calls are asynchronous (the CPU program continues execution after
	// call, but no necessarily before the kernel finishes)
	vectorAdd <<<NUM_BLOCKS, NUM_THREADS >>> (d_a, d_b, d_c, N);

	// Copy sum vector from device to host
	// cudaMemcpy is a synchronous operation, and waits for the prior kernel
	// launch to complete (both go to the default stream in this case).
	// Therefore, this cudaMemcpy acts as both a memcpy and synchronization
	// barrier.
	cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

	// Check result for errors
	verify_result(a, b, c);

	// Free memory on device
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	std::cout << "COMPLETED SUCCESSFULLY\n";

	return 0;
}