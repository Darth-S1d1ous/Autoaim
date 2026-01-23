cmake_minimum_required(VERSION 3.25)

project(Autoaim)

set(CMAKE_CXX_STANDARD 17)

add_executable(Autoaim
	src/app/EntryPoint.cpp
	src/capture/dxgi_capture.cpp
)

target_include_directories(Autoaim PRIVATE src)

target_link_libraries(Autoaim
	d3d11
	dxgi
)