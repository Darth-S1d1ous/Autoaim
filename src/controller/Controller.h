#pragma once
#include <Windows.h>

class Controller {
public:
  // Move mouse to (x, y) and click if isValid is true.
  // Internally tracks a cooldown between clicks (in control-thread ticks).
  void acquireTarget(int x, int y, bool isValid);
  void clickAt(int x, int y);

private:
  int m_targetX = 0;
  int m_targetY = 0;
  bool m_isValid = false;
  int m_coolDown = 0; // ticks remaining before next click is allowed
  static constexpr int CLICK_COOLDOWN_TICKS = 10;
};