cmake_minimum_required(VERSION 3.25)

project(Autoaim)

set(CMAKE_CXX_STANDARD 17)
set(TENSORRT_ROOT ${CMAKE_SOURCE_DIR}/vendor/TensorRT)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

enable_language(CUDA)

add_executable(Autoaim
	src/app/EntryPoint.cpp
	src/capture/dxgi_capture.cpp
	src/log/Log.cpp
)

target_include_directories(Autoaim PRIVATE 
	src
	${CMAKE_SOURCE_DIR}/vendor/spdlog/include
	${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(Autoaim
	d3d11
	dxgi
	${OpenCV_LIBS}
)

add_executable(TestInference
	src/app/TestInference.cpp
	src/detector/Detector.cpp
	src/detector/TrtEngine.cpp
	src/log/Log.cpp
)

target_include_directories(TestInference PRIVATE
	src
	${CMAKE_SOURCE_DIR}/vendor/spdlog/include
	${OpenCV_INCLUDE_DIRS}
	${TENSORRT_ROOT}/include
)

target_link_directories(TestInference PRIVATE
	${TENSORRT_ROOT}/lib
)

target_link_libraries(TestInference
	nvinfer
	nvinfer_plugin
	nvonnxparser
)