#include "capture/dxgi_capture.h"
#include <atomic>
#include <csignal>
#include <thread>
#include <iostream>

std::atomic<bool> running(true);

static void signalHandler(int) {
	running = false;
}

int main()
{
	signal(SIGINT, signalHandler);

	DXGICapture cap;
	if (!cap.init()) {
		return -1;
	}

	while (running) {
		if (cap.capture()) {
			std::cout << "Captured frame "
				<< cap.width() << "x" << cap.height() << std::endl;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(16));
	}
}