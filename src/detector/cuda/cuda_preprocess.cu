#include "detector/cuda_preprocess.h"
#include <cuda_runtime.h>

__global__ void letterBox(const uchar* srcData, const int srcH, const int srcW, uchar* tgtData,
	const int tgtH, const int tgtW, const int rszH, const int rszW, const int startY, const int startX)
{

}

__global__ void process(const uchar* srcData, float* tgtData, const int h, const int w, bool doNormalize = true)
{

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
	letterBox<<<gridSize, blockSize, 0, stream>>> (srcDevData, srcHeight, srcWidth, midDevData, dstHeight, dstWidth, letterBoxH, letterBoxW, startY, startX);
	process<<<gridSize, blockSize, 0, stream >>> (midDevData, dstDevData, dstHeight, dstWidth, doNormalize);

	cudaFreeAsync(srcDevData, stream);
	cudaFreeAsync(midDevData, stream);

	cudaHostUnregister(srcImg.data);
}