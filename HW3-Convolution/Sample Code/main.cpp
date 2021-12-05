/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample implements the same algorithm as the convolutionSeparable
 * CUDA Sample, but without using the shared memory at all.
 * Instead, it uses textures in exactly the same way an OpenGL-based
 * implementation would do.
 * Refer to the "Performance" section of convolutionSeparable whitepaper.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <iostream>
#include <tuple>

#include <helper_functions.h>
#include <helper_cuda.h>

#include "convolutionTexture_common.h"

using std::cout;
using std::cin;
using std::endl;
using std::tuple;
using std::get;

tuple<dim3, int, int> variableAcceptance(int argc, char** argv)
{
	// User Input
	dim3 dimsInputMatrix;
	int maskLength;
	int threadCount;

	if (argc == 4)
	{
		if (atoi(argv[1]) > 0)
			dimsInputMatrix.y = atoi(argv[1]);

		if (atoi(argv[2]) > 0)
			dimsInputMatrix.x = atoi(argv[2]);

		dimsInputMatrix.z = dimsInputMatrix.x * dimsInputMatrix.y;

		if (atoi(argv[3]) > 0)
			maskLength = atoi(argv[3]);

		threadCount = 32;

		cout << "Three Command Line Arguments Accepted." << endl;
	}
	else if (argc == 5)
	{
		if (atoi(argv[1]) > 0)
			dimsInputMatrix.y = atoi(argv[1]);

		if (atoi(argv[2]) > 0)
			dimsInputMatrix.x = atoi(argv[2]);

		dimsInputMatrix.z = dimsInputMatrix.x * dimsInputMatrix.y;

		if (atoi(argv[3]) > 0)
			maskLength = atoi(argv[3]);

		if (atoi(argv[4]) > 0)
			threadCount = atoi(argv[4]);

		cout << "Four Command Line Arguments Accepted." << endl;
	}
	else
	{
		cout << "No/Invalid Command Line Arguments provided. Swapping to user input..." << endl;
		cout << "---------------------------------------------------------------------" << endl;

		cout << "Enter row dimensions for the input matrix: ";
		cin >> dimsInputMatrix.y;

		cout << "Enter column dimensions for the input matrix: ";
		cin >> dimsInputMatrix.x;

		// Total number of Elements in the input matrix
		dimsInputMatrix.z = dimsInputMatrix.x * dimsInputMatrix.y;

		cout << "Enter the length of the mask: ";
		cin >> maskLength;

		cout << "Enter number of threads per block: ";
		cin >> threadCount;

		if (threadCount > 32);
			//threadCount = 32;
	}
	cout << "---------------------------------------------------------------------" << endl;
	cout << "Rows of Input Matrix (Y)\t\t\t: " << dimsInputMatrix.y << endl;
	cout << "Columns of Input Matrix (X)\t\t\t: " << dimsInputMatrix.y << endl;
	cout << "Number of Elements in the Input Matrix (X * Y)\t: " << dimsInputMatrix.z << endl;
	cout << "Length of 1D Mask (K)\t\t\t\t: " << maskLength << endl;
	cout << "Thread count per Block\t\t\t\t: " << threadCount << endl;
	cout << "---------------------------------------------------------------------" << endl;

	tuple<dim3, int, int> output(dimsInputMatrix, maskLength, threadCount);
	return output;
}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	float *h_Mask, *h_Input, *h_Buffer, *h_OutputCPU,	*h_OutputGPU;

	cudaArray *a_Src;
	cudaTextureObject_t texSrc;
	cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float>();

	float *d_Output;

	float gpuTime;

	StopWatchInterface *hTimer = NULL;

	tuple<dim3, int, int> CLA = variableAcceptance(argc, argv); 
	const int imageW = get<0>(CLA).x;
	const int imageH = get<0>(CLA).y;
	const unsigned int iterations = 10;

	const int numElements = get<0>(CLA).z;
	const int maskLeng = get<1>(CLA);


	printf("[%s] - Starting...\n", argv[0]);

	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	sdkCreateTimer(&hTimer);

	printf("Initializing data...\n");
	h_Mask		= (float *)malloc(maskLeng * sizeof(float));
	h_Input     = (float *)malloc(numElements * sizeof(float));
	h_Buffer    = (float *)malloc(numElements * sizeof(float));
	h_OutputCPU = (float *)malloc(numElements * sizeof(float));
	h_OutputGPU = (float *)malloc(numElements * sizeof(float));
	checkCudaErrors(cudaMallocArray(&a_Src, &floatTex, imageW, imageH));
	checkCudaErrors(cudaMalloc((void **)&d_Output, imageW * imageH * sizeof(float)));

	cudaResourceDesc            texRes;
	memset(&texRes,0,sizeof(cudaResourceDesc));

	texRes.resType            = cudaResourceTypeArray;
	texRes.res.array.array    = a_Src;

	cudaTextureDesc             texDescr;
	memset(&texDescr,0,sizeof(cudaTextureDesc));

	texDescr.normalizedCoords	= false;
	texDescr.filterMode			= cudaFilterModeLinear;
	texDescr.addressMode[0]		= cudaAddressModeWrap;
	texDescr.addressMode[1]		= cudaAddressModeWrap;
	texDescr.readMode			= cudaReadModeElementType;

	checkCudaErrors(cudaCreateTextureObject(&texSrc, &texRes, &texDescr, NULL));

	srand(2009);

	for (unsigned int i = 0; i < maskLeng; i++)
	{
		h_Mask[i] = (float)(rand() % 16);
	}

	for (unsigned int i = 0; i < numElements; i++)
	{
		h_Input[i] = (float)(rand() % 16);
	}

	checkCudaErrors(cudaMemcpyToArray(a_Src, 0, 0, h_Input, numElements * sizeof(float), cudaMemcpyHostToDevice));

	printf("Running GPU rows convolution (%u identical iterations)...\n", iterations);
	checkCudaErrors(cudaDeviceSynchronize());
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	for (unsigned int i = 0; i < iterations; i++)
	{
		convolutionRowsGPU(d_Output, a_Src, imageW, imageH, texSrc, h_Mask, maskLeng, get<2>(CLA));
	}

	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&hTimer);
	gpuTime = sdkGetTimerValue(&hTimer) / (float)iterations;
	printf("Average convolutionRowsGPU() time: %f msecs; //%f Mpix/s\n", gpuTime, numElements * 1e-6 / (0.001 * gpuTime));

	//While CUDA kernels can't write to textures directly, this copy is inevitable
	printf("Copying convolutionRowGPU() output back to the texture...\n");
	checkCudaErrors(cudaDeviceSynchronize());
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	checkCudaErrors(cudaMemcpyToArray(a_Src, 0, 0, d_Output, numElements * sizeof(float), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&hTimer);
	gpuTime = sdkGetTimerValue(&hTimer);
	printf("cudaMemcpyToArray() time: %f msecs; //%f Mpix/s\n", gpuTime, numElements * 1e-6 / (0.001 * gpuTime));

	printf("Running GPU columns convolution (%i iterations)\n", iterations);
	checkCudaErrors(cudaDeviceSynchronize());
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	for (int i = 0; i < iterations; i++)
	{
		convolutionColumnsGPU(d_Output, a_Src, imageW, imageH, texSrc, h_Mask, maskLeng, get<2>(CLA));
	}

	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&hTimer);
	gpuTime = sdkGetTimerValue(&hTimer) / (float)iterations;
	printf("Average convolutionColumnsGPU() time: %f msecs; //%f Mpix/s\n", gpuTime, numElements * 1e-6 / (0.001 * gpuTime));

	printf("Reading back GPU results...\n");
	checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output, numElements * sizeof(float), cudaMemcpyDeviceToHost));

	printf("Checking the results...\n");
	printf("...running convolutionRowsCPU()\n");
	convolutionRowsCPU(h_Buffer, h_Input, h_Mask, imageW, imageH, maskLeng/2);

	printf("...running convolutionColumnsCPU()\n");
	convolutionColumnsCPU(h_OutputCPU, h_Buffer, h_Mask, imageW, imageH, maskLeng/2);

	double delta = 0;
	double sum = 0;

	for (unsigned int i = 0; i < numElements; i++)
	{
		sum += h_OutputCPU[i] * h_OutputCPU[i];
		delta += (h_OutputGPU[i] - h_OutputCPU[i]) * (h_OutputGPU[i] - h_OutputCPU[i]);
	}

	double L2norm = sqrt(delta / sum);
	printf("Relative L2 norm: %E\n", L2norm);
	printf("Shutting down...\n");

	checkCudaErrors(cudaFree(d_Output));
	checkCudaErrors(cudaFreeArray(a_Src));
	free(h_OutputGPU);
	free(h_Buffer);
	free(h_Input);
	free(h_Mask);

	sdkDeleteTimer(&hTimer);

	if (L2norm > 1e-6)
	{
		printf("Test failed!\n");
		exit(EXIT_FAILURE);
	}

	printf("Test passed\n");
	exit(EXIT_SUCCESS);
}
