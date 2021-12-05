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


#ifndef CONVOLUTIONTEXTURE_COMMON_H
#define CONVOLUTIONTEXTURE_COMMON_H


#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////
// Reference CPU convolution
////////////////////////////////////////////////////////////////////////////////
extern "C" void convolutionRowsCPU(float *h_Dst, float *h_Src, float *h_Mask, int imageW, int imageH, int maskR);

extern "C" void convolutionColumnsCPU(float *h_Dst, float *h_Src, float *h_Mask, int imageW, int imageH, int maskR);


////////////////////////////////////////////////////////////////////////////////
// GPU texture-based convolution
////////////////////////////////////////////////////////////////////////////////
extern "C" void setConvolutionMask(float *h_Mask);

extern "C" void convolutionRowsGPU(float *d_Dst, cudaArray *a_Src, int imageW, int imageH, cudaTextureObject_t texSrc, const float* mask, const int maskLeng, const int threadCount);

extern "C" void convolutionColumnsGPU(float *d_Dst, cudaArray *a_Src, int imageW, int imageH, cudaTextureObject_t texSrc, const float* mask, const int maskLeng, const int threadCount);

#endif