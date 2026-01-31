#include "detector/Detector.h"
#include "detector/TrtEngine.h"
#include <filesystem>
#include <string>

int main()
{
	std::filesystem::path workspace = std::filesystem::path("D:/Autoaim");
	std::filesystem::path img_path = workspace / "train" / "data" / "images" / "test" / "scene_25" / "000001.jpg";
	cv::Mat img = cv::imread(img_path.string());
	std::vector<cv::Mat> imageBatch = { img };
	std::vector<std::vector<engine::BBox>> results;

	std::filesystem::path enginePath = workspace / "models" / "head_yolov8s.engine";
	engine::YoloEngine yoloEngine{};
	yoloEngine.build(enginePath.string());
	yoloEngine.warmup();
	yoloEngine.infer(imageBatch, results);

	for (int i = 0; i < imageBatch.size(); ++i)
	{
		for (const engine::BBox& bbox : results[i])
		{
			cv::Rect rect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);
			cv::rectangle(imageBatch[i], rect, cv::Scalar(0, 255, 0), 1);
		}
		cv::imshow("Result " + std::to_string(i), imageBatch[i]);
		cv::waitKey(0);
	}
}