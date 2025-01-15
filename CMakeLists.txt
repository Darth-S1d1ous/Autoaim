cmake_minimum_required(VERSION 3.25)

project(Autoaim)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# set your own path to libraries here
set(OpenCV_DIR "E:/Program Files/opencv/build/x64/vc16/lib")
set(Eigen_DIR "E:/Program Files/Eigen3/share/eigen3/cmake")
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

enable_language(CUDA)

include_directories(${CMAKE_SOURCE_DIR}/include
    "E:/Program Files/NVIDIA/cudnn/v9.6/include"
    "E:/Program Files/NVIDIA/TensorRT/TensorRT-10.7.0.23/include"
    "E:/Program Files/Eigen3/include/eigen3"
)

add_executable(Autoaim src/main.cpp
    src/TRTInferenceEngine.cpp
)

target_include_directories(Autoaim PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN_INCLUDE_DIRS}
)

target_link_libraries(Autoaim
    "E:/Program Files/opencv/build/x64/vc16/lib"
)
