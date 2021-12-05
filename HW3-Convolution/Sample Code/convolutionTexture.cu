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


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <helper_cuda.h>
#include "device_launch_parameters.h"
#include <cuda.h>

#include "convolutionTexture_common.h"


////////////////////////////////////////////////////////////////////////////////
// GPU-specific defines
////////////////////////////////////////////////////////////////////////////////
//Maps to a single instruction on G8x / G9x / G10x
#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )

//Use unrolled innermost convolution loop
#define UNROLL_INNER 1

//Round a / b to nearest higher integer value
inline int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Align a to nearest higher multiple of b
inline int iAlignUp(int a, int b)
{
    return (a % b != 0) ? (a - a % b + b) : a;
}

////////////////////////////////////////////////////////////////////////////////
// Convolution kernel and input array storage
////////////////////////////////////////////////////////////////////////////////
__constant__ float c_Kernel[MASK_LENGTH];

extern "C" void setConvolutionKernel(float* h_Kernel)
{
    cudaMemcpyToSymbol(c_Kernel, h_Kernel, MASK_LENGTH * sizeof(float));
}

////////////////////////////////////////////////////////////////////////////////
// Loop unrolling templates, needed for best performance
////////////////////////////////////////////////////////////////////////////////
template<int i> __device__ float convolutionRow(float x, float y, cudaTextureObject_t texSrc)
{
    return tex2D<float>(texSrc, x + (float)(MASK_RADIUS - i), y) * shared_mask[i] + convolutionRow<i - 1>(x, y, texSrc);
}

template<> __device__ float convolutionRow<-1>(float x, float y, cudaTextureObject_t texSrc)
{
    return 0;
}

template<int i> __device__ float convolutionColumn(float x, float y, cudaTextureObject_t texSrc)
{
    return tex2D<float>(texSrc, x, y + (float)(MASK_RADIUS - i)) * shared_mask[i] + convolutionColumn<i - 1>(x, y, texSrc);
}

template<> __device__ float convolutionColumn<-1>(float x, float y, cudaTextureObject_t texSrc)
{
    return 0;
}

////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRowsMask(float *d_Dst, int imageW, int imageH, cudaTextureObject_t texSrc, const float* __restrict__  c_Mask)
{
    extern __shared__ float* shared_mask[];

    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

    if (ix >= imageW || iy >= imageH)
    {
        return;
    }

    if (IMAD(iy, imageW, ix) < imageW * imageH)
    {
        shared_mask[IMAD(iy, imageW, ix)] = c_Mask[IMAD(iy, imageW, ix)];
    }

    float sum = 0;

#if(UNROLL_INNER)
    sum = convolutionRow<2 * MASK_RADIUS>(x, y, texSrc);
#else

    for (int k = -MASK_RADIUS; k <= MASK_RADIUS; k++)
    {
        sum += tex2D<float>(texSrc, x + (float)k, y) * c_Mask[MASK_RADIUS - k];
    }

#endif

    d_Dst[IMAD(iy, imageW, ix)] = sum;
}

extern "C" void convolutionRowsGPU(float *d_Dst, cudaArray *a_Src, int imageW, int imageH, cudaTextureObject_t texSrc, const float* mask, const int threadCount)
{

    dim3 threads(threadCount, threadCount);
    dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));
    size_t SHARMEM = imageW * imageH * sizeof(float);

    convolutionRowsMask <<<blocks, threads, SHARMEM>>> (d_Dst, imageW, imageH, texSrc, mask);
    getLastCudaError("convolutionRowsMask() execution failed\n");
}

////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColumnsMask(float *d_Dst, int imageW, int imageH, cudaTextureObject_t texSrc, const float* __restrict__  c_Mask)
{
    extern __shared__ float* shared_mask[];

    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

    if (ix >= imageW || iy >= imageH)
    {
        return;
    }

    if (IMAD(iy, imageW, ix) < imageW * imageH)
    {
        shared_mask[IMAD(iy, imageW, ix)] = c_Mask[IMAD(iy, imageW, ix)];
    }

    float sum = 0;

#if(UNROLL_INNER)
    sum = convolutionColumn<2 *MASK_RADIUS>(x, y, texSrc);
#else

    for (int k = -MASK_RADIUS; k <= MASK_RADIUS; k++)
    {
        sum += tex2D<float>(texSrc, x, y + (float)k) * c_Mask[MASK_RADIUS - k];
    }

#endif

    d_Dst[IMAD(iy, imageW, ix)] = sum;
}

extern "C" void convolutionColumnsGPU(float *d_Dst, cudaArray *a_Src, int imageW, int imageH, cudaTextureObject_t texSrc, const float* mask, const int threadCount)
{
    dim3 threads(threadCount, threadCount);
    dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));
    size_t SHARMEM = imageW * imageH * sizeof(float);

    convolutionColumnsMask<<<blocks, threads, SHARMEM>>>(d_Dst, imageW, imageH, texSrc, mask);
    getLastCudaError("convolutionColumnsMask() execution failed\n");
}
