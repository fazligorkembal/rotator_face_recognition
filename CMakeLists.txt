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

pkg_check_modules(GSTREAMER QUIET IMPORTED_TARGET gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)

if(GSTREAMER_FOUND)
    set(GSTREAMER_TARGET PkgConfig::GSTREAMER)
    set(GSTREAMER_INCLUDE_DIR_SET ${GSTREAMER_INCLUDE_DIRS})
    message("${ESC}[1;32m[INFO] Found GStreamer via pkg-config: ${GSTREAMER_VERSION}${ESC}[0m")
else()
    find_path(GSTREAMER_INCLUDE_DIR
        NAMES gst/gst.h
        PATH_SUFFIXES gstreamer-1.0
        PATHS /usr/include /usr/local/include
        REQUIRED
    )
    find_path(GSTREAMER_APP_INCLUDE_DIR
        NAMES gst/app/gstappsink.h
        PATH_SUFFIXES gstreamer-1.0
        PATHS /usr/include /usr/local/include
        REQUIRED
    )
    find_path(GLIB_INCLUDE_DIR
        NAMES glib.h
        PATH_SUFFIXES glib-2.0
        PATHS /usr/include /usr/local/include
        REQUIRED
    )
    find_path(GLIB_CONFIG_INCLUDE_DIR
        NAMES glibconfig.h
        PATH_SUFFIXES glib-2.0/include
        PATHS /usr/lib /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/local/lib
        REQUIRED
    )

    find_library(GSTREAMER_LIBRARY NAMES gstreamer-1.0 REQUIRED)
    find_library(GSTREAMER_APP_LIBRARY NAMES gstapp-1.0 REQUIRED)
    find_library(GSTREAMER_VIDEO_LIBRARY NAMES gstvideo-1.0 REQUIRED)
    find_library(GSTREAMER_BASE_LIBRARY NAMES gstbase-1.0 REQUIRED)
    find_library(GOBJECT_LIBRARY NAMES gobject-2.0 REQUIRED)
    find_library(GLIB_LIBRARY NAMES glib-2.0 REQUIRED)

    add_library(gstreamer_manual INTERFACE)
    target_include_directories(gstreamer_manual INTERFACE
        ${GSTREAMER_INCLUDE_DIR}
        ${GSTREAMER_APP_INCLUDE_DIR}
        ${GLIB_INCLUDE_DIR}
        ${GLIB_CONFIG_INCLUDE_DIR}
    )
    target_link_libraries(gstreamer_manual INTERFACE
        ${GSTREAMER_LIBRARY}
        ${GSTREAMER_APP_LIBRARY}
        ${GSTREAMER_VIDEO_LIBRARY}
        ${GSTREAMER_BASE_LIBRARY}
        ${GOBJECT_LIBRARY}
        ${GLIB_LIBRARY}
    )

    set(GSTREAMER_TARGET gstreamer_manual)
    set(GSTREAMER_INCLUDE_DIR_SET
        ${GSTREAMER_INCLUDE_DIR}
        ${GSTREAMER_APP_INCLUDE_DIR}
        ${GLIB_INCLUDE_DIR}
        ${GLIB_CONFIG_INCLUDE_DIR}
    )
    message("${ESC}[1;33m[INFO] pkg-config GStreamer not found, using manual include/lib discovery${ESC}[0m")
endif()
message("${ESC}[1;32m[INFO] Found spdlog: ${spdlog_DIR}${ESC}[0m")
message("${ESC}[1;32m[INFO] Found CUDA Toolkit: ${CUDAToolkit_BIN_DIR}${ESC}[0m")
message("${ESC}[1;32m[INFO] Found OpenCV: ${OpenCV_DIR}${ESC}[0m")

include_directories(
    include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${GSTREAMER_INCLUDE_DIR_SET}
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
    ${GSTREAMER_TARGET}
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
