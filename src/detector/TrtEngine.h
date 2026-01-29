#pragma once

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>

namespace engine
{
	class TrtLogger : public nvinfer1::ILogger 
	{
	public:
		void log(Severity severity, const char* msg) noexcept;

	};

	struct BBox 
	{
		int x1, y1, x2, y2;
		float score;
		int classId;
	};

	class TrtEngineBase
	{
	public:
		using ImageBatch = std::vector<cv::Mat>;
		TrtEngineBase();
		virtual ~TrtEngineBase();

		TrtEngineBase(const TrtEngineBase&) = delete;
		TrtEngineBase& operator=(const TrtEngineBase&) = delete;

		void build(const std::string&);

	protected:
		int maxBatchSize;

		const char* inputName;
		const char* outputName;
		
		std::unique_ptr<TrtLogger> logger;
		std::unique_ptr<nvinfer1::IRuntime> runtime;
		std::unique_ptr<nvinfer1::ICudaEngine> engine;
		nvinfer1::Dims inputDims, outputDims;

		void deserializeEngineFromFile(const std::string&);
		static void letterBox(const cv::Mat&, cv::Mat&, cv::Size);
		static BBox letterBox2Original(const BBox&, const cv::Size, const cv::Size);
		virtual void preprocessBase(const ImageBatch&, float *, bool doNormalize = true);
	};

	class YoloEngine : public TrtEngineBase
	{
	private:
		const float CLS_THRES;
		const float NMS_THRES;

		void postprocess(const std::vector<float>&, std::vector<std::vector<BBox>>&, std::vector<cv::Size>&);

	public:
		YoloEngine(float cls_thres = 0.3, float nms_thres = 0.4);
		void infer(ImageBatch&, std::vector<std::vector<BBox>>&);
		void warmup();

	};

} // namespace engine