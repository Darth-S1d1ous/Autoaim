#include "TRTInferenceEngine.hpp"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <string>
#include <vector>
#include <omp.h>

using namespace engine;

void Logger::log(Severity severity, const char* msg) noexcept
{
    switch (severity) {
    case Severity::kINTERNAL_ERROR:
        std::cerr << "[INTERNAL_ERROR]: " << msg << std::endl;
        break;
    case Severity::kERROR:
        std::cerr << "[ERROR]: " << msg << std::endl;
        break;
    default:
        break;
    }
}

/************* TrtEngineBase *************/

TrtEngineBase::TrtEngineBase() : 
    logger(new Logger()),
    runtime(nvinfer1::createInferRuntime(*logger))
{}

TrtEngineBase::~TrtEngineBase()
{
    engine.reset();
    runtime.reset();
}

void TrtEngineBase::deserializeEngineFromFile(const std::string& enginePath)
{
    std::ifstream engineFile(enginePath, std::ios::binary);
    if (!engineFile || !engineFile.is_open()) {
        logger->log(Logger::Severity::kERROR, "Error: Failed to open engine file");
        throw std::runtime_error("Error: Failed to open engine file");
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

    inputName = engine->getIOTensorName(0);
    outputName = engine->getIOTensorName(1);
    inputDims = engine->getTensorShape(inputName.c_str());
    outputDims = engine->getTensorShape(outputName.c_str());
    maxBatchSize = inputDims.d[0];
}

void TrtEngineBase::letterBox(const cv::Mat& img, cv::Mat& letterBoxed, cv::Size targetSize){
    float scale = std::min(static_cast<float>(targetSize.width) / img.cols, static_cast<float>(targetSize.height) / img.rows);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(), scale, scale, cv::INTER_LINEAR);
    // Use copyMakeBorder to pad the image
    int top = (targetSize.height - resized.rows) / 2;
    int bottom = targetSize.height - resized.rows - top;
    int left = (targetSize.width - resized.cols) / 2;
    int right = targetSize.width - resized.cols - left;
    cv::copyMakeBorder(resized, letterBoxed, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
}

BBox TrtEngineBase::letterBox2Original(const BBox& bbox, const cv::Size originalSize, const cv::Size boxSize){
    float scale = std::min(static_cast<float>(boxSize.width) / originalSize.width, static_cast<float>(boxSize.height) / originalSize.height);
    float dx = (boxSize.width - scale * originalSize.width) / 2;
    float dy = (boxSize.height - scale * originalSize.height) / 2;

    BBox originalBbox;
    originalBbox.x1 = (bbox.x1 - dx) / scale;
    originalBbox.y1 = (bbox.y1 - dy) / scale;
    originalBbox.x2 = (bbox.x2 - dx) / scale;
    originalBbox.y2 = (bbox.y2 - dy) / scale;
    originalBbox.score = bbox.score;
    originalBbox.classId = bbox.classId;

    return originalBbox;
}

void TrtEngineBase::preprocessBase(const ImageBatch &imgs, float *proceessed, bool doNormalize){
    assert(imgs.size() <= maxBatchSize);
}


/************* YoloEngine *************/

YoloEngine::YoloEngine(float cls_thres, float nms_thres):CLS_THRES(cls_thres), NMS_THRES(nms_thres){}

