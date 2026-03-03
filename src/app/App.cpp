#include "App.h"
#include "log/Log.h"
#include <chrono>
#include <vector>

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
void AutoAimer::init(const Config &cfg) {
  const std::string enginePath = cfg.get("engine_path");
  const float clsThres = cfg.getFloat("cls_threshold", 0.35f);
  const float nmsThres = cfg.getFloat("nms_threshold", 0.45f);

  m_capturer = std::make_unique<DXGICapture>();
  m_detector = std::make_unique<Detector>();
  m_controller = std::make_unique<Controller>();

  if (!m_capturer->init())
    throw std::runtime_error("[App] DXGICapture init failed");

  m_detector->init(enginePath);

  CORE_INFO("[App] All components initialized. Engine: {}", enginePath);
}

// ---------------------------------------------------------------------------
// Capture thread  (Producer → image channel)
// ---------------------------------------------------------------------------
void AutoAimer::captureWorker(SharedData *shared) {
  if (!m_capturer->ok()) {
    CORE_ERROR("[Capture] DXGICapture is not initialized!");
    shared->isRunning = false;
    return;
  }

  CORE_INFO("[Capture] Thread started.");

  while (shared->isRunning) {
    if (m_capturer->capture()) {
      std::lock_guard<std::mutex> lock(shared->imageMutex);
      m_capturer->getLatestFrame(shared->latestImage);
      shared->hasNewImage = true;
      shared->imageCV.notify_one();
    } else {
      // No new frame yet; yield briefly to avoid spinning
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  }

  CORE_INFO("[Capture] Thread exiting.");
}

// ---------------------------------------------------------------------------
// Inference thread  (Consumer image → Producer target)
// ---------------------------------------------------------------------------
void AutoAimer::inferenceWorker(SharedData *shared, int aimOffsetY) {
  CORE_INFO("[Inference] Thread started.");

  while (shared->isRunning) {
    cv::Mat currentFrame;

    // Wait for a new frame
    {
      std::unique_lock<std::mutex> lock(shared->imageMutex);
      shared->imageCV.wait(
          lock, [shared] { return shared->hasNewImage || !shared->isRunning; });
      if (!shared->isRunning)
        break;

      shared->latestImage.copyTo(currentFrame);
      shared->hasNewImage = false;
    }

    // Run YOLO detection
    std::vector<cv::Mat> imageBatch = {currentFrame};
    std::vector<std::vector<engine::BBox>> results;
    m_detector->detect(imageBatch, results);

    // Pick the highest-confidence detection as target
    bool found = false;
    int bestX = 0, bestY = 0;
    float bestScore = 0.f;

    if (!results.empty()) {
      for (const auto &bbox : results[0]) {
        if (bbox.score > bestScore) {
          bestScore = bbox.score;
          // Aim at top-center of head bbox + configurable Y offset
          bestX = (bbox.x1 + bbox.x2) / 2;
          bestY = bbox.y1 + aimOffsetY;
          found = true;
        }
      }
    }

    // Publish target
    {
      std::lock_guard<std::mutex> lock(shared->targetMutex);
      shared->target.valid = found;
      if (found) {
        shared->target.x = bestX;
        shared->target.y = bestY;
        CORE_TRACE("[Inference] Target at ({}, {}), score={:.2f}", bestX, bestY,
                   bestScore);
      }
    }
  }

  CORE_INFO("[Inference] Thread exiting.");
}

// ---------------------------------------------------------------------------
// Control thread  (Consumer target → mouse output)
// ---------------------------------------------------------------------------
void AutoAimer::controlWorker(SharedData *shared) {
  CORE_INFO("[Control] Thread started.");

  while (shared->isRunning) {
    {
      std::lock_guard<std::mutex> lock(shared->targetMutex);
      m_controller->acquireTarget(shared->target.x, shared->target.y,
                                  shared->target.valid);
    }
    // Poll at ~60 Hz — fast enough to track but not wasteful
    std::this_thread::sleep_for(std::chrono::milliseconds(16));
  }

  CORE_INFO("[Control] Thread exiting.");
}

// ---------------------------------------------------------------------------
// Run — wires everything together
// ---------------------------------------------------------------------------
void AutoAimer::run() {
  SharedData shared;
  const int aimOffsetY =
      10; // default; ideally stored from init(cfg) in a member

  CORE_INFO("[App] Starting pipeline threads...");

  std::thread tCapture([this, &shared] { captureWorker(&shared); });
  std::thread tInference(
      [this, &shared, aimOffsetY] { inferenceWorker(&shared, aimOffsetY); });
  std::thread tControl([this, &shared] { controlWorker(&shared); });

  // Block until isRunning goes false (set by signal handler in EntryPoint)
  while (shared.isRunning)
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Wake the inference thread in case it's waiting on a frame
  shared.imageCV.notify_all();

  tCapture.join();
  tInference.join();
  tControl.join();

  CORE_INFO("[App] All threads joined. Goodbye.");
}