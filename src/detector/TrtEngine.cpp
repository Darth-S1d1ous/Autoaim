#include "TrtEngine.h"
#include "cuda_preprocess.h"
#include <fstream>

namespace engine
{
	void TrtLogger::log(Severity severity, const char* msg) noexcept
	{
		// suppress info-level messages
		switch (severity) {
		case Severity::kINTERNAL_ERROR:
			std::cerr << "[INTERNAL_ERROR]: " << msg << std::endl;
			break;
		case Severity::kERROR:
			std::cerr << "[ERROR]: " << msg << std::endl;
			break;
		case Severity::kWARNING:
			std::cerr << "[WARNING]: " << msg << std::endl;
			break;
		default:
			break;
		}
	}

	TrtEngineBase::TrtEngineBase() :
		logger(new TrtLogger()),
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

	// For a GPU version, go to the .cu file
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

	YoloEngine::YoloEngine(float cls_thres, float nms_thres):
		CLS_THRES(cls_thres),
		NMS_THRES(nms_thres)
	{

	}

	/*
	bboxes: empty before inference; filled after inference
	*/
	void YoloEngine::infer(ImageBatch& imgs, std::vector<std::vector<BBox>>& bboxes)
	{
		if (imgs.size() > maxBatchSize)
		{
			logger->log(TrtLogger::Severity::kWARNING, "Warning: Input batch size exceeds max batch size. Truncating input batch.");
			imgs.resize(maxBatchSize);
		}

		auto context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
		cudaStream_t stream;
		cudaStreamCreate(&stream);

		// input: batch, channel, height, width; output: 
		size_t n_inputElem = inputDims.d[0] * inputDims.d[1] * inputDims.d[2] * inputDims.d[3];
		size_t n_outputElem = outputDims.d[0] * outputDims.d[1] * outputDims.d[2];
		size_t inputSize = n_inputElem * sizeof(float);
		size_t outputSize = n_outputElem * sizeof(float);

		// Allocate memory for input and output tensors
		void* buffers[2];
		cudaMalloc(&buffers[0], inputSize);
		cudaMalloc(&buffers[1], outputSize);
		context->setTensorAddress(inputName, static_cast<float*>(buffers[0]));
		context->setTensorAddress(outputName, static_cast<float*>(buffers[1]));

		preprocessBase(imgs, static_cast<float*>(buffers[0]), true);

		context->enqueueV3(stream);

		std::vector<float> output(n_outputElem);
		cudaMemcpyAsync(output.data(), buffers[1], outputSize, cudaMemcpyDeviceToHost, stream);

		cudaFreeAsync(buffers[0], stream);
		cudaFreeAsync(buffers[1], stream);
		cudaStreamSynchronize(stream);
		cudaStreamDestroy(stream);

		// post-process
		std::vector<cv::Size> originalSizes(imgs.size(), cv::Size());
		for (int i = 0; i < imgs.size(); ++i)
		{
			if (!imgs[i].empty())
			{
				originalSizes[i] = imgs[i].size();
			}
		}
		postprocess(output, bboxes, originalSizes);
	}

	void YoloEngine::postprocess(const std::vector<float>& output, std::vector<std::vector<BBox>>& results, std::vector<cv::Size>& originalSizes)
	{
		size_t n_rows = outputDims.d[1]; // 4 + num_classes
		size_t n_cols = outputDims.d[2]; // num_boxes

		results.clear();
		results.resize(originalSizes.size());

		// For each image in the batch
		for (int bid = 0; bid < originalSizes.size(); ++bid)
		{
			if (originalSizes[bid].empty())
			{
				results[bid] = {};
				continue;
			}

			// Get the start pointer of the output for this image
			const float* data = output.data() + bid * n_rows * n_cols;
			std::vector<cv::Rect> boxes;
			std::vector<float> scores;
			std::vector<int> classes;

			for (int j = 0; j < n_cols; ++j)
			{
				float max_conf = 0;
				int max_id;

				for (int i = 4; i < n_rows; ++i)
				{
					int idx = i * n_cols + j;
					if (data[idx] > max_conf)
					{
						max_conf = data[idx];
						max_id = i - 4;
					}
				}
				if (max_conf > CLS_THRES)
				{
					float x = data[j];
					float y = data[j + n_cols];
					float w = data[j + 2 * n_cols];
					float h = data[j + 3 * n_cols];
					float left = x - w / 2;
					float top = y - h / 2;
					boxes.emplace_back(cv::Rect(left, top, w, h));
					scores.emplace_back(max_conf);
					classes.emplace_back(max_id);
				}
			}

			// NMS
			std::vector<int> indices;
			cv::dnn::NMSBoxes(boxes, scores, 0.0, NMS_THRES, indices);
			std::vector<BBox> bboxes;
			for (int id : indices)
			{
				BBox bbox;
				bbox.x1 = boxes[id].x;
				bbox.y1 = boxes[id].y;
				bbox.x2 = boxes[id].x + boxes[id].width;
				bbox.y2 = boxes[id].y + boxes[id].height;
				bbox.score = scores[id];
				bbox.classId = classes[id];

				bbox = letterBox2Original(bbox, originalSizes[bid], cv::Size(inputDims.d[3], inputDims.d[2]));
				bboxes.emplace_back(bbox);
			}
			results[bid] = bboxes;
		}
	}

	void YoloEngine::warmup()
	{
		// Generate zero images for warmup
		ImageBatch imgs(maxBatchSize, cv::Mat::zeros(inputDims.d[2], inputDims.d[3], CV_8UC3));
		std::vector<std::vector<BBox>> bboxes;
		for (int i = 0; i < 5; ++i)
		{
			infer(imgs, bboxes);
		}
	}
}