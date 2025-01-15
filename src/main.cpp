#include <iostream>
#include "TRTInferenceEngine.hpp"

using namespace engine;

int main()
{
    YoloEngine poseEngine(0.5, 0.5);
    poseEngine.build("../models/yolo11n-pose.pt");
    std::cout << "Hello, World!" << std::endl;
    return 0;
}