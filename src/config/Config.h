#pragma once
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>


// Simple key=value config file loader.
// Lines starting with '#' or ';' are comments.
// Blank lines are ignored.
// Example:
//   engine_path = D:/Autoaim/models/head_yolov8s.engine
//   cls_threshold = 0.35
//   nms_threshold = 0.45

class Config {
public:
  explicit Config(const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open())
      throw std::runtime_error("Cannot open config file: " + path);

    std::string line;
    while (std::getline(file, line)) {
      // Strip carriage return
      if (!line.empty() && line.back() == '\r')
        line.pop_back();

      // Skip comments and blank lines
      if (line.empty() || line[0] == '#' || line[0] == ';')
        continue;

      auto eq = line.find('=');
      if (eq == std::string::npos)
        continue;

      std::string key = trim(line.substr(0, eq));
      std::string value = trim(line.substr(eq + 1));
      m_data[key] = value;
    }
  }

  const std::string &get(const std::string &key) const {
    auto it = m_data.find(key);
    if (it == m_data.end())
      throw std::runtime_error("Config key not found: " + key);
    return it->second;
  }

  std::string get(const std::string &key, const std::string &defaultVal) const {
    auto it = m_data.find(key);
    return it != m_data.end() ? it->second : defaultVal;
  }

  float getFloat(const std::string &key, float defaultVal) const {
    auto it = m_data.find(key);
    if (it == m_data.end())
      return defaultVal;
    return std::stof(it->second);
  }

  int getInt(const std::string &key, int defaultVal) const {
    auto it = m_data.find(key);
    if (it == m_data.end())
      return defaultVal;
    return std::stoi(it->second);
  }

private:
  std::unordered_map<std::string, std::string> m_data;

  static std::string trim(std::string s) {
    const char *ws = " \t";
    s.erase(0, s.find_first_not_of(ws));
    s.erase(s.find_last_not_of(ws) + 1);
    return s;
  }
};
