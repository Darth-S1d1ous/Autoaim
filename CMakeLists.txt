cmake_minimum_required(VERSION 3.25)

project(Autoaim)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# set your own path to libraries here
find_package(OpenCV REQUIRED)
find_package(Eigen3)

enable_language(CUDA)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(Autoaim src/main.cpp
    src/TRTInferenceEngine.cpp
)

target_include_directories(Autoaim PUBLIC
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)

target_link_libraries(Autoaim
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    ${EIGEN3_LIBRARIES}
    nvinfer
    nvonnxparser
)
