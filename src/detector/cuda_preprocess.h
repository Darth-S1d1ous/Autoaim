#pragma once

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

/*
doNormalize: 
*/
void cuda_preprocess(cudaStream_t& stream, const cv::Mat& srcImg, float* dstData, const int dstHeight, const int dstWidth, bool doNormalize = true);