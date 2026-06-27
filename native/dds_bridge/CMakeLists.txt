cmake_minimum_required(VERSION 3.16)
project(visual_events_dds_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(UNITREE_SDK_ROOT "" CACHE PATH "Unitree SDK2 install root")
set(VIDEO_DDS_PUBLISHER_DIR "" CACHE PATH "video_dds_publisher source root")
option(VISUAL_EVENTS_DDS_BRIDGE_BUILD_PROBE "Build the Unitree/CameraFrame DDS type-support probe" ON)
option(VISUAL_EVENTS_DDS_BRIDGE_FULL_BRIDGE "Compile generated Visual Events Head/Gaze DDS type support into the probe" OFF)
set(VISUAL_EVENTS_GENERATED_DDS_DIR "" CACHE PATH "Generated Visual Events Head/Gaze CycloneDDS C++ output directory")
set(VISUAL_EVENTS_HEAD_STATE_HEADER "" CACHE FILEPATH "Generated head_state_v1.hpp")
set(VISUAL_EVENTS_HEAD_STATE_SOURCE "" CACHE FILEPATH "Generated head_state_v1.cpp")
set(VISUAL_EVENTS_GAZE_TARGET_HEADER "" CACHE FILEPATH "Generated gaze_target_v1.hpp")
set(VISUAL_EVENTS_GAZE_TARGET_SOURCE "" CACHE FILEPATH "Generated gaze_target_v1.cpp")

add_library(visual_events_dds_bridge_contract STATIC
    src/bridge_contract.cpp
)
target_include_directories(visual_events_dds_bridge_contract PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_options(visual_events_dds_bridge_contract PRIVATE -Wall -Wextra -Wpedantic)

add_library(visual_events_dds_bridge_abi STATIC
    src/bridge_abi.cpp
)
target_include_directories(visual_events_dds_bridge_abi PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_options(visual_events_dds_bridge_abi PRIVATE -Wall -Wextra -Wpedantic)

add_executable(visual_events_dds_bridge
    src/runtime_main.cpp
)
target_link_libraries(visual_events_dds_bridge PRIVATE
    visual_events_dds_bridge_abi
)
target_compile_options(visual_events_dds_bridge PRIVATE -Wall -Wextra -Wpedantic)

add_executable(visual_events_dds_bridge_abi_harness
    src/abi_harness_main.cpp
)
target_link_libraries(visual_events_dds_bridge_abi_harness PRIVATE
    visual_events_dds_bridge_abi
)
target_compile_options(visual_events_dds_bridge_abi_harness PRIVATE -Wall -Wextra -Wpedantic)

if (VISUAL_EVENTS_DDS_BRIDGE_BUILD_PROBE)
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

    set(VISUAL_EVENTS_GENERATED_DDS_SOURCES "")
    if (VISUAL_EVENTS_DDS_BRIDGE_FULL_BRIDGE)
        if (NOT VISUAL_EVENTS_GENERATED_DDS_DIR)
            message(FATAL_ERROR "VISUAL_EVENTS_GENERATED_DDS_DIR is required when VISUAL_EVENTS_DDS_BRIDGE_FULL_BRIDGE=ON")
        endif()
        if (NOT VISUAL_EVENTS_HEAD_STATE_HEADER)
            set(VISUAL_EVENTS_HEAD_STATE_HEADER "${VISUAL_EVENTS_GENERATED_DDS_DIR}/head_state_v1.hpp")
        endif()
        if (NOT VISUAL_EVENTS_HEAD_STATE_SOURCE)
            set(VISUAL_EVENTS_HEAD_STATE_SOURCE "${VISUAL_EVENTS_GENERATED_DDS_DIR}/head_state_v1.cpp")
        endif()
        if (NOT VISUAL_EVENTS_GAZE_TARGET_HEADER)
            set(VISUAL_EVENTS_GAZE_TARGET_HEADER "${VISUAL_EVENTS_GENERATED_DDS_DIR}/gaze_target_v1.hpp")
        endif()
        if (NOT VISUAL_EVENTS_GAZE_TARGET_SOURCE)
            set(VISUAL_EVENTS_GAZE_TARGET_SOURCE "${VISUAL_EVENTS_GENERATED_DDS_DIR}/gaze_target_v1.cpp")
        endif()

        foreach(GENERATED_DDS_INPUT
            "${VISUAL_EVENTS_HEAD_STATE_HEADER}"
            "${VISUAL_EVENTS_HEAD_STATE_SOURCE}"
            "${VISUAL_EVENTS_GAZE_TARGET_HEADER}"
            "${VISUAL_EVENTS_GAZE_TARGET_SOURCE}"
        )
            if (NOT EXISTS "${GENERATED_DDS_INPUT}")
                message(FATAL_ERROR "Missing generated Visual Events DDS type support input: ${GENERATED_DDS_INPUT}")
            endif()
        endforeach()

        list(APPEND VISUAL_EVENTS_GENERATED_DDS_SOURCES
            "${VISUAL_EVENTS_HEAD_STATE_SOURCE}"
            "${VISUAL_EVENTS_GAZE_TARGET_SOURCE}"
        )
    endif()

    list(APPEND CMAKE_PREFIX_PATH
        "${UNITREE_SDK_ROOT}"
        "${UNITREE_SDK_ROOT}/lib/cmake/unitree_sdk2"
    )
    find_package(unitree_sdk2 REQUIRED)

    add_executable(visual_events_dds_bridge_probe
        src/probe_main.cpp
        "${CAMERA_FRAME_SOURCE}"
        ${VISUAL_EVENTS_GENERATED_DDS_SOURCES}
    )
    target_include_directories(visual_events_dds_bridge_probe PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${VIDEO_DDS_PUBLISHER_DIR}/include"
    )
    if (VISUAL_EVENTS_DDS_BRIDGE_FULL_BRIDGE)
        target_include_directories(visual_events_dds_bridge_probe PRIVATE
            "${VISUAL_EVENTS_GENERATED_DDS_DIR}"
        )
        target_compile_definitions(visual_events_dds_bridge_probe PRIVATE
            VISUAL_EVENTS_DDS_BRIDGE_FULL_BRIDGE=1
        )
    endif()
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
endif()
