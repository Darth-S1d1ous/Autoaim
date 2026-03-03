#include "Controller.h"

void Controller::acquireTarget(int x, int y, bool isValid) {
  m_targetX = x;
  m_targetY = y;
  m_isValid = isValid;

  if (!m_isValid)
    return;

  if (m_coolDown > 0) {
    --m_coolDown;
    return;
  }

  // Move cursor to the target and click
  clickAt(m_targetX, m_targetY);
  m_coolDown = CLICK_COOLDOWN_TICKS;
}

void Controller::clickAt(int x, int y) {
  SetCursorPos(x, y);

  INPUT inputs[2] = {};
  inputs[0].type = INPUT_MOUSE;
  inputs[0].mi.dwFlags = MOUSEEVENTF_LEFTDOWN;

  inputs[1].type = INPUT_MOUSE;
  inputs[1].mi.dwFlags = MOUSEEVENTF_LEFTUP;

  SendInput(2, inputs, sizeof(INPUT));
}