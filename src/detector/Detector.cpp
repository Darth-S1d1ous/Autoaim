#include "Detector.h"
#include "TrtEngine.h"

Detector::Detector()
{
}
Detector::~Detector()
{
}

void Detector::init(const std::string& enginePath)
{
	yoloEngine_ = std::make_unique<engine::YoloEngine>();
	yoloEngine_->build(enginePath);
}

bool Detector::detect(std::vector<cv::Mat>& imageBatch, std::vector<std::vector<engine::BBox>>& results)
{
	yoloEngine_->infer(imageBatch, results);

	return true;
}