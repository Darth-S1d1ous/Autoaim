#pragma once

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeBase.h>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <string>
#include <vector>
#include <Eigen/Dense>

namespace engine{
    class Logger : public nvinfer1::ILogger{
        void log(Severity severity, const char* msg) noexcept;
    }; 

    struct BBox{
        int x1, y1, x2, y2;
        float score;
    };

    struct KeyPoints{
        Eigen::MatrixXf keypoints;
        float score;
    };

    class TrtEngineBase{
        public:
            using ImageBatch = std::vector<cv::Mat>;

        protected:
            int maxBatchSize;
            std::string inputName, outputName;
            std::unique_ptr<Logger> logger;
            std::unique_ptr<nvinfer1::IRuntime> runtime;
            std::unique_ptr<nvinfer1::ICudaEngine> engine;
            nvinfer1::Dims inputDims, outputDims;

            void deserializeEngineFromFile(const std::string& enginePath);
            static void letterBox(const cv::Mat& src, cv::Mat& dst, const cv::Size& dstSize);
            static BBox letterBox2Original(const BBox& letterBox, const cv::Size& srcSize, const cv::Size& dstSize);
            virtual void preprocessBase();
        
        public:
            TrtEngineBase();
            virtual ~TrtEngineBase();

            TrtEngineBase(const TrtEngineBase&) = delete;
            TrtEngineBase& operator=(const TrtEngineBase&) = delete;

            void build(const std::string& enginePath);
    };

    class YoloEngine : public TrtEngineBase{
        private:
            const float CLS_THRES;
            const float NMS_THRES;

            void postprocess(const std::vector<cv::Mat>& inputImages, std::vector<std::vector<BBox>>& outputBBoxes, std::vector<cv::Size>&);
        public:
            YoloEngine(float clsThres, float nmsThres);
            void infer(const ImageBatch&, std::vector<std::vector<BBox>>&, std::vector<std::vector<KeyPoints>>&);
            void warmup();
    }; 
}