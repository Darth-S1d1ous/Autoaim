#pragma once
#include "TrtEngine.h"
#include <string>

class Detector
{
public:
	Detector();
	~Detector();

	void init(const std::string& enginePath);
	bool detect(std::vector<cv::Mat>& imageBatch, std::vector<std::vector<engine::BBox>>& results);

private:
	std::unique_ptr<engine::YoloEngine> yoloEngine_;
};