#include "detector/cuda_preprocess.h"
#include <cuda_runtime.h>
#include <cstdint>

__global__ void letterBox(const uchar* srcData, const int srcH, const int srcW, uchar* tgtData,
	const int tgtH, const int tgtW, const int rszH, const int rszW, const int startY, const int startX)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int idx = ix + iy * tgtW;
	int idx3 = idx * 3;

	if (ix > tgtW || iy > tgtH) return;
	// gray region on target image
	if (iy < startY || iy >(startY + rszH - 1)) {
		tgtData[idx3] = 128;
		tgtData[idx3 + 1] = 128;
		tgtData[idx3 + 2] = 128;
		return;
	}
	if (ix < startX || ix >(startX + rszW - 1)) {
		tgtData[idx3] = 128;
		tgtData[idx3 + 1] = 128;
		tgtData[idx3 + 2] = 128;
		return;
	}

	float scaleY = (float)rszH / (float)srcH;
	float scaleX = (float)rszW / (float)srcW;

	// reproject pixel pos in tgt image to a continuous pos in src image
	// pixel center alignment
	float beforeX = float(ix - startX + 0.5) / scaleX - 0.5;
	float beforeY = float(iy - startY + 0.5) / scaleY - 0.5;
	// Reprojected pixel might be in a box, calc the 4 vertices
	int topY = static_cast<int>(beforeY);
	int bottomY = topY + 1;
	int leftX = static_cast<int>(beforeX);
	int rightX = leftX + 1;
	// Calc the decimal part
	float u = beforeX - leftX;
	float v = beforeY - topY;

	if (topY >= srcH - 1 && leftX >= srcW - 1) 
	{
		for (int k = 0; k < 3; k++)
		{
			tgtData[idx3 + k] = (1 - u) * (1 - v) * srcData[(leftX + topY * srcW) * 3 + k];
		}
	}
	else if (topY >= srcH - 1)  // 最后一行
	{
		for (int k = 0; k < 3; k++)
		{
			tgtData[idx3 + k]
				= (1. - u) * (1. - v) * srcData[(leftX + topY * srcW) * 3 + k]
				+ (u) * (1. - v) * srcData[(rightX + topY * srcW) * 3 + k];
		}
	}
	else if (leftX >= srcW - 1)  // 最后一列
	{
		for (int k = 0; k < 3; k++)
		{
			tgtData[idx3 + k]
				= (1. - u) * (1. - v) * srcData[(leftX + topY * srcW) * 3 + k]
				+ (1. - u) * (v)*srcData[(leftX + bottomY * srcW) * 3 + k];
		}
	}
	else  // 非最后一行或最后一列情况
	{
		for (int k = 0; k < 3; k++)
		{
			tgtData[idx3 + k]
				= (1. - u) * (1. - v) * srcData[(leftX + topY * srcW) * 3 + k]
				+ (u) * (1. - v) * srcData[(rightX + topY * srcW) * 3 + k]
				+ (1. - u) * (v)*srcData[(leftX + bottomY * srcW) * 3 + k]
				+ u * v * srcData[(rightX + bottomY * srcW) * 3 + k];
		}
	}
}

// HWC to CHW with optional normalization
__global__ void process(const uchar* srcData, float* tgtData, const int h, const int w, bool doNormalize = true)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int idx = ix + iy * w;
	int idx3 = idx * 3;

	if (ix < w && iy < h)
	{
		tgtData[idx] = (float)srcData[idx3 + 2];
		tgtData[idx + h * w] = (float)srcData[idx3 + 1];
		tgtData[idx + 2 * h * w] = (float)srcData[idx3 + 0];

		if (doNormalize)
		{
			tgtData[idx] /= 255.0f;
			tgtData[idx + h * w] /= 255.0f;
			tgtData[idx + 2 * h * w] /= 255.0f;
		}
	}
}

void cuda_preprocess(cudaStream_t& stream, const cv::Mat& srcImg, float* dstDevData, const int dstHeight, const int dstWidth, bool doNormalize)
{
	int srcHeight = srcImg.rows;
	int srcWidth = srcImg.cols;
	int srcElements = srcHeight * srcWidth * 3;
	int dstElements = dstHeight * dstWidth * 3;
	int letterBoxH, letterBoxW, startX, startY;
	float scale = std::min(static_cast<float>(dstWidth) / srcWidth, static_cast<float>(dstHeight) / srcHeight);
	letterBoxH = static_cast<int>(scale * srcHeight);
	letterBoxW = static_cast<int>(scale * srcWidth);

	// Horizontal and vertical margin size
	startY = (dstHeight - letterBoxH) / 2;
	startX = (dstWidth - letterBoxW) / 2;

	// source images data on device
	uchar *srcDevData, *midDevData;
	cudaMallocAsync((void**)&midDevData, sizeof(uchar) * dstElements, stream);
	cudaHostRegister(srcImg.data, sizeof(uchar) * srcElements, cudaHostRegisterMapped);
	cudaHostGetDevicePointer(&srcDevData, srcImg.data, 0);

	dim3 blockSize(32, 32);
	dim3 gridSize((dstWidth + blockSize.x - 1) / blockSize.x, (dstHeight + blockSize.y - 1) / blockSize.y);
	// srcDevData -> midDevData
	letterBox<<<gridSize, blockSize, 0, stream>>> (srcDevData, srcHeight, srcWidth, midDevData, dstHeight, dstWidth, letterBoxH, letterBoxW, startY, startX);
	// midDevData -> dstDevData
	process<<<gridSize, blockSize, 0, stream >>> (midDevData, dstDevData, dstHeight, dstWidth, doNormalize);

	cudaFreeAsync(srcDevData, stream);
	cudaFreeAsync(midDevData, stream);

	cudaHostUnregister(srcImg.data);
}