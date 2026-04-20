cmake_minimum_required(VERSION 3.18)
project(RFR CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
enable_language(CUDA)

find_package(spdlog REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)

string(ASCII 27 ESC)
message("${ESC}[1;32m[INFO] Found spdlog: ${spdlog_DIR}${ESC}[0m")
message("${ESC}[1;32m[INFO] Found CUDA Toolkit: ${CUDAToolkit_BIN_DIR}${ESC}[0m")
message("${ESC}[1;32m[INFO] Found OpenCV: ${OpenCV_DIR}${ESC}[0m")

include_directories(
    include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(main
    src/main_cuda_nms.cpp
    src/detection_model_inference_helper.cpp
    src/detection_model_inference_helper.cu
)

add_executable(detection_model_converter
    src/converters/detection_model_converter.cpp
)

set(COMMON_LIBS
    spdlog::spdlog
    nvinfer
    nvinfer_plugin
    nvonnxparser
    CUDA::cudart
    ${OpenCV_LIBS}
)

target_link_libraries(main PRIVATE ${COMMON_LIBS})
target_link_libraries(detection_model_converter PRIVATE ${COMMON_LIBS})