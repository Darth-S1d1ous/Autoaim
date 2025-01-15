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
    runtime(nvinfer1::createInferRuntime(*logger)),
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
        logger->log(Severity::kINTERNAL_ERROR, "Failed to open engine file: " + enginePath);
        throw std::runtime_error("Error: Failed to open engine file: " + enginePath);
    }

    engineFile.seekg(0, engineFile.end);
    size_t fileSize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);

    std::unique_ptr<char[]> engineData(new char[fileSize]);
    engineFile.read(engineData.get(), fileSize);
    engineFile.close();

    engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engineData.get(), fileSize, nullptr));
}

void TrtEngineBase::build(const std::string& modelPath)
{
    deserializeEngineFromFile(modelPath);

    inputName = engine->getIOTensorName(0);
    outputName = engine->getIOTensorName(1);
    inputDims = engine->getTensorShape(inputName.c_str());
    outputDims = engine->getTensorShape(outputName.c_str());
    maxBatchSize = engine->getMaxBatchSize();
}

/************* YoloEngine *************/

YoloEngine::YoloEngine(float cls_thres, float nms_thres):CLS_THRES(cls_thres), NMS_THRES(nms_thres){}

