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
    src/identification_model_inference_helper.cpp
    src/identification_model_inference_helper.cu
)

set(COMMON_LIBS
    spdlog::spdlog
    nvinfer
    nvinfer_plugin
    nvonnxparser
    CUDA::cudart
    CUDA::cublas
    ${OpenCV_LIBS}
)

target_link_libraries(main PRIVATE ${COMMON_LIBS})

add_executable(modeldetection_model_converter src/converters/detection_model_converter.cpp)
target_link_libraries(modeldetection_model_converter PRIVATE ${COMMON_LIBS})

add_executable(modelidentification_model_converter src/converters/identification_model_converter.cpp)
target_link_libraries(modelidentification_model_converter PRIVATE ${COMMON_LIBS})


option(LOGRESULTS "Write detection debug outputs under build/logresults" OFF)
if(LOGRESULTS)
    target_compile_definitions(main PRIVATE LOGRESULTS)
    message("${ESC}[1;33m[DEBUG] LOGRESULTS enabled${ESC}[0m")
endif()

option(DISPLAYRESULTS "Display min-box-length filter results on the input image" OFF)
if(DISPLAYRESULTS)
    target_compile_definitions(main PRIVATE DISPLAYRESULTS)
    message("${ESC}[1;33m[DEBUG] DISPLAYRESULTS enabled${ESC}[0m")
endif()
