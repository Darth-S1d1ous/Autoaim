#include <iostream>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include "TRTInferenceEngine.hpp"

using namespace engine;

int main()
{
    YoloEngine poseEngine(0.5, 0.5);
    poseEngine.build("../models/yolo11n-pose.pt");
    std::cout << "Hello, World!" << std::endl;
    return 0;
}