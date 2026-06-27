cmake_minimum_required(VERSION 3.16)
project(visual_events_dds_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(UNITREE_SDK_ROOT "" CACHE PATH "Unitree SDK2 install root")
set(VIDEO_DDS_PUBLISHER_DIR "" CACHE PATH "video_dds_publisher source root")

if (NOT UNITREE_SDK_ROOT)
    message(FATAL_ERROR "UNITREE_SDK_ROOT is required")
endif()

if (NOT VIDEO_DDS_PUBLISHER_DIR)
    message(FATAL_ERROR "VIDEO_DDS_PUBLISHER_DIR is required")
endif()

set(CAMERA_FRAME_HEADER "${VIDEO_DDS_PUBLISHER_DIR}/include/unitree_camera/msg/dds/CameraFrame_.hpp")
set(CAMERA_FRAME_SOURCE "${VIDEO_DDS_PUBLISHER_DIR}/src/CameraFrame_.cpp")

if (NOT EXISTS "${CAMERA_FRAME_HEADER}")
    message(FATAL_ERROR "Missing CameraFrame_ header: ${CAMERA_FRAME_HEADER}")
endif()

if (NOT EXISTS "${CAMERA_FRAME_SOURCE}")
    message(FATAL_ERROR "Missing CameraFrame_ source: ${CAMERA_FRAME_SOURCE}")
endif()

list(APPEND CMAKE_PREFIX_PATH
    "${UNITREE_SDK_ROOT}"
    "${UNITREE_SDK_ROOT}/lib/cmake/unitree_sdk2"
)
find_package(unitree_sdk2 REQUIRED)

add_library(visual_events_dds_bridge_contract STATIC
    src/bridge_contract.cpp
)
target_include_directories(visual_events_dds_bridge_contract PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_options(visual_events_dds_bridge_contract PRIVATE -Wall -Wextra -Wpedantic)

add_executable(visual_events_dds_bridge_probe
    src/probe_main.cpp
    "${CAMERA_FRAME_SOURCE}"
)
target_include_directories(visual_events_dds_bridge_probe PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${VIDEO_DDS_PUBLISHER_DIR}/include"
)
target_link_libraries(visual_events_dds_bridge_probe PRIVATE
    visual_events_dds_bridge_contract
    unitree_sdk2
    -Wl,--no-as-needed
    ddsc
    ddscxx
    -Wl,--as-needed
)
target_compile_options(visual_events_dds_bridge_probe PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(visual_events_dds_bridge_probe PROPERTIES
    BUILD_RPATH "${UNITREE_SDK_ROOT}/lib"
    INSTALL_RPATH "${UNITREE_SDK_ROOT}/lib"
)
