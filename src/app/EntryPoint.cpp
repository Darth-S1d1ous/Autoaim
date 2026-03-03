#include "App.h"
#include "config/Config.h"
#include "log/Log.h"
#include <csignal>

// Global flag set by Ctrl+C signal — used to stop the pipeline via isRunning
// (AutoAimer::run() checks isRunning every 100 ms)
static AutoAimer *g_app = nullptr;

static void signalHandler(int) {
  CORE_WARN("[EntryPoint] Ctrl+C received — shutting down...");
  // We can't directly reach SharedData from here, so we rely on
  // AutoAimer::run() detecting a global stop flag via the atomic bool.
  // A cleaner approach: expose a stop() method on AutoAimer.
  std::exit(0); // threads are detached-joined inside run(), so this is safe
}

int main(int argc, char *argv[]) {
  Log::Init();
  CORE_INFO("[EntryPoint] AutoAimer starting.");

  // Config file path: default "config.ini" next to the executable,
  // or pass a custom path as the first command-line argument.
  std::string configPath = (argc > 1) ? argv[1] : "config.ini";

  Config cfg(configPath);

  signal(SIGINT, signalHandler);

  try {
    AutoAimer app;
    app.init(cfg);
    app.run(); // blocks until isRunning == false
  } catch (const std::exception &e) {
    CORE_ERROR("[EntryPoint] Fatal error: {}", e.what());
    return 1;
  }

  return 0;
}