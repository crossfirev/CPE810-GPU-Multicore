/*
*	File Name:	cuda2DConvolution_ConstMem.cu
*	Language:	CUDA C/C++
*	Desc:		This program implement a 2D convolution routine using constant in CUDA C/C++.
*
*	Author:		Matthew Lepis
*	Date:		11/5/2021
*/

// C++ Includes
#include <iostream>

// CUDA Includes

#include <helper_functions.h>
#include <helper_cuda.h>

// Namespace statements
using std::cout;
using std::cin;
using std::endl;
using std::pair;

// Macro Definitions
#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )

void Clear();
inline int iDivUp(int a, int b);

// Pulled from CUDA sample: "convolutionTexture"; with the addition of shared memory.
__global__ void convolutionRowsGPUKernel(float* d_Dst, int matrixCol_x, int matrixRow_y, float* d_mask, pair<int, int> mask_len_rad, cudaTextureObject_t texSrc)
{
	extern __shared__ float shared_mask[];

	if (threadIdx.x < mask_len_rad.first)
		shared_mask[threadIdx.x] = d_mask[threadIdx.x];
	__syncthreads();

	const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	const float  x = (float)ix + 0.5f;
	const float  y = (float)iy + 0.5f;

	if (ix >= matrixCol_x || iy >= matrixRow_y)
		return;

	float sum = 0;

	for (int k = -mask_len_rad.second; k <= mask_len_rad.second; k++)
		sum += tex2D<float>(texSrc, x + (float)k, y) * shared_mask[mask_len_rad.second - k];


	d_Dst[IMAD(iy, matrixCol_x, ix)] = sum;
}
__global__ void convolutionColsGPUKernel(float* d_Dst, int matrixCol_x, int matrixRow_y, float* d_mask, pair<int, int> mask_len_rad, cudaTextureObject_t texSrc)
{
	extern __shared__ float shared_mask[];

	if (threadIdx.x < mask_len_rad.first)
		shared_mask[threadIdx.x] = d_mask[threadIdx.x];
	__syncthreads();

	const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	const float  x = (float)ix + 0.5f;
	const float  y = (float)iy + 0.5f;

	if (ix >= matrixCol_x || iy >= matrixRow_y)
		return;

	float sum = 0;

	for (int k = -mask_len_rad.second; k <= mask_len_rad.second; k++)
		sum += tex2D<float>(texSrc, x, y + (float)k) * shared_mask[mask_len_rad.second - k];


	d_Dst[IMAD(iy, matrixCol_x, ix)] = sum;
}

// Pulled from CUDA sample: "convolutionTexture"; with the addition of variable threads.
void convolutionRowsGPU(float* d_Dst, int matrixCol_x, int matrixRow_y, float* d_mask, pair<int, int> mask_len_rad, cudaTextureObject_t texSrc, size_t SHARMEM, int threadCount) // IMPLEMENT VARIABLE THREADS. ------------------------------------------------------------------------------
{
	dim3 threads(threadCount, threadCount);
	dim3 blocks(iDivUp(matrixCol_x, threads.x), iDivUp(matrixRow_y, threads.y));

	convolutionRowsGPUKernel <<<blocks, threads, SHARMEM>>> (d_Dst, matrixCol_x, matrixRow_y, d_mask, mask_len_rad, texSrc);
	getLastCudaError("convolutionRowsGPUKernel() execution failed\n");
}
void convolutionColsGPU(float* d_Dst, int matrixCol_x, int matrixRow_y, float* d_mask, pair<int, int> mask_len_rad, cudaTextureObject_t texSrc, size_t SHARMEM, int threadCount) // IMPLEMENT VARIABLE THREADS. ------------------------------------------------------------------------------
{
	dim3 threads(threadCount, threadCount);
	dim3 blocks(iDivUp(matrixCol_x, threads.x), iDivUp(matrixRow_y, threads.y));

	convolutionColsGPUKernel <<<blocks, threads, SHARMEM >>> (d_Dst, matrixCol_x, matrixRow_y, d_mask, mask_len_rad, texSrc);
	getLastCudaError("convolutionColsGPUKernel() execution failed\n");
}

// Pulled from CUDA sample: "convolutionTexture".
void convolutionRowsCPU(float* h_Dst, float* h_Src, float* h_Mask, int imageW, int imageH, int maskR)
{
	for (int y = 0; y < imageH; y++)
		for (int x = 0; x < imageW; x++)
		{
			float sum = 0;

			for (int k = -maskR; k <= maskR; k++)
			{
				int d = x + k;

				if (d < 0) d = 0;

				if (d >= imageW) d = imageW - 1;

				sum += h_Src[y * imageW + d] * h_Mask[maskR - k];
			}

			h_Dst[y * imageW + x] = sum;
		}
}
void convolutionColsCPU(float* h_Dst, float* h_Src, float* h_Mask, int imageW, int imageH, int maskR)
{
	for (int y = 0; y < imageH; y++)
		for (int x = 0; x < imageW; x++)
		{
			float sum = 0;

			for (int k = -maskR; k <= maskR; k++)
			{
				int d = y + k;

				if (d < 0) d = 0;

				if (d >= imageH) d = imageH - 1;

				sum += h_Src[d * imageW + x] * h_Mask[maskR - k];
			}

			h_Dst[y * imageW + x] = sum;
		}
}

// Original Implimentation; Inspired from the CUDA sample: "convolutionTexture".
int do2DConvolution(dim3 inputDims, const int maskDim, const int threadCount, int iterations)
{
	printf("Starting...\n");

	// Input matrix data size
	size_t x_col_bytes	= inputDims.x * sizeof(float);
	size_t y_row_bytes	= inputDims.y * sizeof(float);
	size_t z_full_bytes	= inputDims.z * sizeof(float);

	// Mask data size
	size_t mask_bytes	= maskDim * sizeof(int);
	pair<int, int> mask_len_rad(maskDim, maskDim / 2);
		// first	= Mask Length
		// second	= Mask Radius

	// Timing Metrics
	float gpuTime, cpuTime;
	StopWatchInterface* hTimer = NULL;
	sdkCreateTimer(&hTimer);

	// Host Memory Declarations
	float *h_Mask, *h_Input, *h_OutputCPU, *h_OutputGPU, *h_Buffer;

	// Allocating Host memory space for float ptrs.
	checkCudaErrors(cudaHostAlloc(&h_Mask, mask_bytes, cudaHostAllocDefault));
	checkCudaErrors(cudaHostAlloc(&h_Mask, mask_bytes, cudaHostAllocDefault));
	checkCudaErrors(cudaHostAlloc(&h_Input, z_full_bytes, cudaHostAllocDefault));
	checkCudaErrors(cudaHostAlloc(&h_OutputCPU, z_full_bytes, cudaHostAllocDefault));
	checkCudaErrors(cudaHostAlloc(&h_OutputGPU, z_full_bytes, cudaHostAllocDefault));
	checkCudaErrors(cudaHostAlloc(&h_Buffer, z_full_bytes, cudaHostAllocDefault));

	// Device Memory Declarations
	float *d_OutputGPU, *d_Mask;

	// Texture Memory Stuff
	cudaArray *a_Src;
	cudaTextureObject_t texSrc;
	cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float>();

	// Allocating Device memory space for float ptrs and CUDA arrays.
	checkCudaErrors(cudaMalloc(&d_OutputGPU, z_full_bytes));
	checkCudaErrors(cudaMalloc(&d_Mask, mask_bytes));

	checkCudaErrors(cudaMallocArray(&a_Src, &floatTex, inputDims.x, inputDims.y));

	// Setting up textures
	cudaResourceDesc texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));

	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = a_Src;

	cudaTextureDesc texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	texDescr.normalizedCoords = false;
	texDescr.filterMode = cudaFilterModeLinear;
	texDescr.addressMode[0] = cudaAddressModeWrap;
	texDescr.addressMode[1] = cudaAddressModeWrap;
	texDescr.readMode = cudaReadModeElementType;

	checkCudaErrors(cudaCreateTextureObject(&texSrc, &texRes, &texDescr, NULL));

	// Setting the random seed
	srand(2009);


	// Filling mask and input matrix
	printf("Initializing data...\n\n");
	for (unsigned int i = 0; i < mask_len_rad.first; i++)
		h_Mask[i] = (float)(rand() % 16);

	for (unsigned int i = 0; i < inputDims.z; i++)
		h_Input[i] = (float)(rand() % 16);

	checkCudaErrors(cudaMemcpyToArray(a_Src, 0, 0, h_Input, z_full_bytes, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Mask, h_Mask, mask_bytes, cudaMemcpyHostToDevice));

	printf("Running GPU rows convolution (%u identical iterations)...\n", iterations);
	checkCudaErrors(cudaDeviceSynchronize());

	// Starting Timer for convolutionRowsGPU()
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	// Running the row convolution filter 'iteration' number of times
	for (unsigned int i = 0; i < iterations; i++)
		convolutionRowsGPU(d_OutputGPU, inputDims.x, inputDims.y, d_Mask, mask_len_rad, texSrc, mask_bytes, threadCount);
	checkCudaErrors(cudaDeviceSynchronize());

	// Stopping Timer and calculating single-run time for print
	sdkStopTimer(&hTimer);
	gpuTime = sdkGetTimerValue(&hTimer) / (float)iterations;
	printf("Average convolutionRowsGPU() time: %f msecs; //%f Mpix/s\n\n", gpuTime, inputDims.z * 1e-6 / (0.001 * gpuTime));

	// Copying the result of the row convolution to the texture memory for column convolution. On host, since kernels can't write to texture memory.
	printf("Copying convolutionRowsGPU() output back to the texture...\n");
	checkCudaErrors(cudaDeviceSynchronize());

	// Starting Timer for cudaMemcpyToArray()
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	checkCudaErrors(cudaMemcpyToArray(a_Src, 0, 0, d_OutputGPU, inputDims.z * sizeof(float), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaDeviceSynchronize());

	// Stopping Timer and calculating copy time for print
	sdkStopTimer(&hTimer);
	gpuTime = sdkGetTimerValue(&hTimer);
	printf("cudaMemcpyToArray() time: %f msecs; //%f Mpix/s\n\n", gpuTime, inputDims.z * 1e-6 / (0.001 * gpuTime));
	
	printf("Running GPU columns convolution (%i iterations)\n", iterations);
	checkCudaErrors(cudaDeviceSynchronize());

	// Starting Timer for convolutionColsGPU()
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	// Running the col convolution filter 'iteration' number of times
	for (int i = 0; i < iterations; i++)
		convolutionColsGPU(d_OutputGPU, inputDims.x, inputDims.y, d_Mask, mask_len_rad, texSrc, mask_bytes, threadCount);
	checkCudaErrors(cudaDeviceSynchronize());

	// Stopping Timer and calculating single-run time for print
	sdkStopTimer(&hTimer);
	gpuTime = sdkGetTimerValue(&hTimer) / (float)iterations;
	printf("Average convolutionColumnsGPU() time: %f msecs; //%f Mpix/s\n\n", gpuTime, inputDims.z * 1e-6 / (0.001 * gpuTime));

	printf("Reading back GPU results...\n");
	checkCudaErrors(cudaMemcpy(h_OutputGPU, d_OutputGPU, inputDims.z * sizeof(float), cudaMemcpyDeviceToHost));

	printf("Checking the results...\n");
	printf("\t...running convolutionRowsCPU()\n");

	// Starting Timer for convolutionRowsCPU()
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	convolutionRowsCPU(h_Buffer, h_Input, h_Mask, inputDims.x, inputDims.y, mask_len_rad.second);
	sdkStopTimer(&hTimer);
	cpuTime = sdkGetTimerValue(&hTimer);
	printf("Average convolutionRowsCPU() time: %f msecs; //%f Mpix/s\n\n", cpuTime, inputDims.z * 1e-6 / (0.001 * cpuTime));

	printf("\t...running convolutionColumnsCPU()\n");
	// Starting Timer for convolutionRowsCPU()
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	convolutionColsCPU(h_OutputCPU, h_Buffer, h_Mask, inputDims.x, inputDims.y, mask_len_rad.second);
	sdkStopTimer(&hTimer);
	cpuTime = sdkGetTimerValue(&hTimer);
	printf("Average convolutionColsCPU() time: %f msecs; //%f Mpix/s\n\n", cpuTime, inputDims.z * 1e-6 / (0.001 * cpuTime));

	// Calculating the difference between the GPU and the CPU runs.
	double delta = 0;
	double sum = 0;

	for (unsigned int i = 0; i < inputDims.z; i++)
	{
		sum += h_OutputCPU[i] * h_OutputCPU[i];
		delta += (h_OutputGPU[i] - h_OutputCPU[i]) * (h_OutputGPU[i] - h_OutputCPU[i]);
	}
	double L2norm = sqrt(delta / sum);
	printf("Relative L2 norm: %E\n\n", L2norm);


	printf("Shutting down...\n\n");

	checkCudaErrors(cudaFree(d_OutputGPU));
	checkCudaErrors(cudaFree(d_Mask));
	checkCudaErrors(cudaFreeArray(a_Src));

	checkCudaErrors(cudaFreeHost(h_OutputGPU));
	checkCudaErrors(cudaFreeHost(h_Buffer));
	checkCudaErrors(cudaFreeHost(h_Input));
	checkCudaErrors(cudaFreeHost(h_Mask));

	sdkDeleteTimer(&hTimer);
	bool exitVal;
	if (L2norm > 1e-6)
	{
		printf("Test failed!\n");
		exitVal = EXIT_FAILURE;
	}
	else
	{
		printf("Test passed.\n");
		exitVal = EXIT_SUCCESS;
	}
	return exitVal;
}
int main(int argc, char** argv)
{
	// User Input
	dim3 dimsInputMatrix;
	int maskLength;
	int threadCount = 32;
	int iterations = 10;

	if (argc == 4)
	{
		if (atoi(argv[1]) > 0)
			dimsInputMatrix.y = atoi(argv[1]);

		if (atoi(argv[2]) > 0)
			dimsInputMatrix.x = atoi(argv[2]);

		if (atoi(argv[3]) > 0)
			maskLength = atoi(argv[3]);

		cout << "Three Command Line Arguments Accepted." << endl;
	}
	else if (argc == 5)
	{
		if (atoi(argv[1]) > 0)
			dimsInputMatrix.y = atoi(argv[1]);

		if (atoi(argv[2]) > 0)
			dimsInputMatrix.x = atoi(argv[2]);

		if (atoi(argv[3]) > 0)
			maskLength = atoi(argv[3]);

		if (atoi(argv[4]) > 0)
			threadCount = atoi(argv[4]);

		cout << "Four Command Line Arguments Accepted." << endl;
	}
	else if (argc == 6)
	{
		if (atoi(argv[1]) > 0)
			dimsInputMatrix.y = atoi(argv[1]);

		if (atoi(argv[2]) > 0)
			dimsInputMatrix.x = atoi(argv[2]);

		if (atoi(argv[3]) > 0)
			maskLength = atoi(argv[3]);

		if (atoi(argv[4]) > 0)
			threadCount = atoi(argv[4]);

		if (atoi(argv[4]) > 0)
			iterations = atoi(argv[5]);

		cout << "Five Command Line Arguments Accepted." << endl;
	}
	else
	{
		cout << "No/Invalid Command Line Arguments provided. Swapping to user input..." << endl;
		cout << "---------------------------------------------------------------------" << endl;

		cout << "Enter row dimensions for the input matrix: ";
		cin >> dimsInputMatrix.y;

		Clear();

		cout << "Enter row dimensions for the input matrix: ";
		cin >> dimsInputMatrix.x;

		Clear();

		cout << "Enter the length of the mask: ";
		cin >> maskLength;

		Clear();

		cout << "Enter number of threads per block: ";
		cin >> threadCount;

		Clear();

		cout << "Enter number of iterations to run for each GPU kernel call: ";
		cin >> iterations;

		Clear();
	}

	// Total number of Elements in the input matrix
	dimsInputMatrix.z = dimsInputMatrix.x * dimsInputMatrix.y;

	if (threadCount > 32)
		threadCount = 32;
	
	cout << "---------------------------------------------------------------------\n";
	cout << "| Rows of Input Matrix (Y)\t\t\t\t: "					<< dimsInputMatrix.y << "\n";
	cout << "| Columns of Input Matrix (X)\t\t\t\t: "				<< dimsInputMatrix.y << "\n";
	cout << "| Number of Elements in the Input Matrix (X * Y)\t: "	<< dimsInputMatrix.z << "\n";
	cout << "| Length of 1D Mask (K)\t\t\t\t\t: "					<< maskLength << "\n";
	cout << "| Thread count per Block\t\t\t\t: "					<< threadCount << "\n";
	cout << "| Test Iterations:\t\t\t\t\t: "						<< iterations << "\n";
	cout << "---------------------------------------------------------------------" << endl;

	// Computation & Validation
	return do2DConvolution(dimsInputMatrix, maskLength, threadCount, iterations);
}

// Misc. Functions
void Clear()
{
#if defined _WIN32
	system("cls");
	//clrscr(); // including header file : conio.h
#elif defined (__LINUX__) || defined(__gnu_linux__) || defined(__linux__)
	system("clear");
	//std::cout<< u8"\033[2J\033[1;1H"; //Using ANSI Escape Sequences 
#elif defined (__APPLE__)
	system("clear");
#endif
}
inline int iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}