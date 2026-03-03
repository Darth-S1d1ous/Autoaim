cmake_minimum_required(VERSION 3.25)

project(Autoaim)

set(CMAKE_CXX_STANDARD 17)
set(TENSORRT_ROOT ${CMAKE_SOURCE_DIR}/vendor/TensorRT)
message(STATUS "TensorRT path: ${TENSORRT_ROOT}")

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

enable_language(CUDA)

# -----------------------------------------------------------------------
# Autoaim — capture-only smoke test (no TensorRT dependency)
# -----------------------------------------------------------------------
add_executable(TestCapture
	src/app/TestCapture.cpp
	src/capture/dxgi_capture.cpp
	src/log/Log.cpp
)

target_include_directories(TestCapture PRIVATE 
	src
	${CMAKE_SOURCE_DIR}/vendor/spdlog/include
	${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(TestCapture
	d3d11
	dxgi
	${OpenCV_LIBS}
)

# -----------------------------------------------------------------------
# TestInference — stand-alone TRT inference test
# -----------------------------------------------------------------------
add_executable(TestInference
    src/app/TestInference.cpp
    src/detector/Detector.cpp
    src/detector/TrtEngine.cpp
    src/detector/cuda/cuda_preprocess.cu
    src/log/Log.cpp
)

target_include_directories(TestInference PRIVATE
    src
    ${CMAKE_SOURCE_DIR}/vendor/spdlog/include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TENSORRT_ROOT}/include
)

target_link_directories(TestInference PRIVATE
    ${TENSORRT_ROOT}/lib
)

target_link_libraries(TestInference
    ${OpenCV_LIBS}
    CUDA::cudart
    nvinfer_10
    nvinfer_plugin_10
    nvonnxparser_10
)

# -----------------------------------------------------------------------
# AutoaimApp — full pipeline: capture + inference + control
# -----------------------------------------------------------------------
add_executable(AutoaimApp
    src/app/EntryPoint.cpp
    src/app/App.cpp
    src/capture/dxgi_capture.cpp
    src/controller/Controller.cpp
    src/detector/Detector.cpp
    src/detector/TrtEngine.cpp
    src/detector/cuda/cuda_preprocess.cu
    src/log/Log.cpp
)

target_include_directories(AutoaimApp PRIVATE
    src
    ${CMAKE_SOURCE_DIR}/vendor/spdlog/include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TENSORRT_ROOT}/include
)

target_link_directories(AutoaimApp PRIVATE
    ${TENSORRT_ROOT}/lib
)

target_link_libraries(AutoaimApp
    d3d11
    dxgi
    ${OpenCV_LIBS}
    CUDA::cudart
    nvinfer_10
    nvinfer_plugin_10
    nvonnxparser_10
    delayimp                     # required for /DELAYLOAD
)

# Delay-load CUDA & TensorRT DLLs so they are NOT loaded at process start.
# This prevents their DllMain hooks from interfering with DXGI DuplicateOutput.
# The DLLs will be loaded on first function call (i.e. when Detector::init runs,
# AFTER DXGICapture::init has already succeeded).
target_link_options(AutoaimApp PRIVATE
    "/DELAYLOAD:cudart64_13.dll"
    "/DELAYLOAD:nvinfer_10.dll"
    "/DELAYLOAD:nvinfer_plugin_10.dll"
    "/DELAYLOAD:nvonnxparser_10.dll"
)

# Copy config.ini next to the AutoaimApp binary at build time
add_custom_command(TARGET AutoaimApp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/config.ini"
        "$<TARGET_FILE_DIR:AutoaimApp>/config.ini"
    COMMENT "Copying config.ini to output directory"
)