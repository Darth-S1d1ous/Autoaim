#pragma once

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>

namespace engine
{
	struct BBox 
	{
		int x1, y1, x2, y2;
		float score;
		int classId;
	};

	class TrtEngineBase
	{
	public:
		TrtEngineBase();
		virtual ~TrtEngineBase();

		TrtEngineBase(const TrtEngineBase&) = delete;
		TrtEngineBase& operator=(const TrtEngineBase&) = delete;

		void build(const std::string&);

	protected:
		int maxBatchSize;
		
		std::unique_ptr<nvinfer1::IRuntime> runtime;
		std::unique_ptr<nvinfer1::ICudaEngine> engine;
	};

	class YoloEngine : public TrtEngineBase
	{
	private:
		const float CLS_THRES;
		const float NMS_THRES;

		// void postprocess(const std::vector<float>&, std::vector<std::vector<BBox>>&, std::vector<cv::Size>&);
	};
} // namespace engine