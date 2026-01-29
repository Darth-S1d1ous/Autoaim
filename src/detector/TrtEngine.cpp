#include "TrtEngine.h"
#include "cuda_preprocess.h"
#include <fstream>

namespace engine
{
	TrtEngineBase::TrtEngineBase() :
		logger(new TrtLogger())
		runtime(nvinfer1::createInferRuntime(*logger))
	{
	}

	TrtEngineBase::~TrtEngineBase()
	{
		engine.reset();
		runtime.reset();
	}

	void TrtEngineBase::deserializeEngineFromFile(const std::string& enginePath)
	{
		std::ifstream engineFile(enginePath, std::ios::binary);
		if (!engineFile)
		{
			logger->log(TrtLogger::Severity::kERROR, "Error: Unable to open engine file");
			throw std::runtime_error("Error: Unable to open engine file");
		}
		engineFile.seekg(0, engineFile.end);
		size_t fileSize = engineFile.tellg();
		engineFile.seekg(0, engineFile.beg);

		std::unique_ptr<char[]> engineData(new char[fileSize]);
		engineFile.read(engineData.get(), fileSize);
		engineFile.close();

		engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engineData.get(), fileSize));
	}

	void TrtEngineBase::build(const std::string& modelPath)
	{
		deserializeEngineFromFile(modelPath);

		// Each input / output tensor has a name in TensorRT
		inputName = engine->getIOTensorName(0);
		outputName = engine->getIOTensorName(1);
		inputDims = engine->getTensorShape(inputName);
		outputDims = engine->getTensorShape(outputName);
		maxBatchSize = inputDims.d[0];
	}

	void TrtEngineBase::letterBox(const cv::Mat& img, cv::Mat& letterBoxed, cv::Size targetSize)
	{
		float scale = std::min(static_cast<float>(targetSize.width) / img.cols, static_cast<float>(targetSize.height) / img.rows);

		cv::Mat resized;
		cv::resize(img, resized, cv::Size(), scale, scale, cv::INTER_LINEAR);

		// Pad the image. The following are padding sizes, not absolute height or width
		int top = (targetSize.height - resized.rows) / 2;
		int bottom = targetSize.height - top - resized.rows;
		int left = (targetSize.width - resized.cols) / 2;
		int right = targetSize.width - left - resized.cols;
		cv::copyMakeBorder(resized, letterBoxed, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	}

	BBox TrtEngineBase::letterBox2Original(const BBox& bbox, cv::Size originalSize, cv::Size boxSize)
	{
		float scale = std::min(static_cast<float>(boxSize.width) / originalSize.width, static_cast<float>(boxSize.height) / originalSize.height);
		// Calc horizontal and vertical padding sizes
		// boxSize is the size of the whole letterbox
		float dx = (boxSize.width - scale * originalSize.width) / 2;
		float dy = (boxSize.height - scale * originalSize.height) / 2;

		BBox originalBBox;
		originalBBox.x1 = (bbox.x1 - dx) / scale;
		originalBBox.y1 = (bbox.y1 - dy) / scale;
		originalBBox.x2 = (bbox.x2 - dx) / scale;
		originalBBox.y2 = (bbox.y2 - dy) / scale;
		originalBBox.score = bbox.score;
		originalBBox.classId = bbox.classId;

		return originalBBox;
	}

	void TrtEngineBase::preprocessBase(const ImageBatch& imgs, float* processed, bool doNormalize)
	{
		assert(imgs.size() <= maxBatchSize);

		std::vector<cudaStream_t> streams(imgs.size());
		for (int i = 0; i < imgs.size(); ++i)
		{
			cudaStreamCreate(&streams[i]);
		}

		for (int i = 0; i < imgs.size(); ++i)
		{
			if (imgs[i].empty()) continue;
			// Assume NCHW format: batchSize, channel, height, width
			int offset = i * inputDims.d[1] * inputDims.d[2] * inputDims.d[3];
			cuda_preprocess(streams[i], imgs[i], processed + offset, inputDims.d[2], inputDims.d[3], doNormalize);
		}

		for (int i = 0; i < imgs.size(); ++i)
		{
			cudaStreamSynchronize(streams[i]);
			cudaStreamDestroy(streams[i]);
		}
	}
}