#pragma once
#include "capture/dxgi_capture.h"
#include "config/Config.h"
#include "controller/Controller.h"
#include "detector/Detector.h"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>


// Shared state passed between the three worker threads.
// Image channel:  captureWorker produces, inferenceWorker consumes
// (condition_variable). Target channel: inferenceWorker produces, controlWorker
// consumes (mutex poll). Lifetime:       isRunning == false shuts every thread
// down gracefully.
struct SharedData {
  // --- Image channel (capture → inference) ---
  cv::Mat latestImage;
  bool hasNewImage = false;
  std::mutex imageMutex;
  std::condition_variable imageCV;

  // --- Target channel (inference → control) ---
  struct Target {
    int x = 0;
    int y = 0;
    bool valid = false;
  } target;
  std::mutex targetMutex;

  // --- Lifetime ---
  std::atomic<bool> isRunning{true};

  // Non-copyable (mutex members are not copyable anyway)
  SharedData() = default;
  SharedData(const SharedData &) = delete;
  SharedData &operator=(const SharedData &) = delete;
};

class AutoAimer {
public:
  void init(const Config &cfg);
  void run();

private:
  void captureWorker(SharedData *shared);
  void inferenceWorker(SharedData *shared, int aimOffsetY);
  void controlWorker(SharedData *shared);

  std::unique_ptr<DXGICapture> m_capturer;
  std::unique_ptr<Detector> m_detector;
  std::unique_ptr<Controller> m_controller;
};