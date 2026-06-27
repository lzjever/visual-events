#pragma once

#include "visual_events/dds_bridge/bridge_abi.hpp"

#include "unitree_camera/msg/dds/CameraFrame_.hpp"

#ifdef VISUAL_EVENTS_DDS_BRIDGE_FULL_BRIDGE
#include "gaze_target_v1.hpp"
#include "head_state_v1.hpp"
#endif

#include <cstdint>

namespace visual_events {
namespace dds_bridge {

CameraJpegFrame CameraFrameToAbi(
    const unitree_camera::msg::dds_::CameraFrame_& frame,
    int64_t received_monotonic_ns);

#ifdef VISUAL_EVENTS_DDS_BRIDGE_FULL_BRIDGE
HeadStateFrame HeadStateToAbi(
    const visual_events::msg::dds_::HeadStateV1_& frame,
    int64_t received_monotonic_ns,
    double stationary_yaw_vel_rad_s = 0.03,
    double stationary_pitch_vel_rad_s = 0.03);

visual_events::msg::dds_::GazeTargetV1_ GazeTargetFrameToDds(const GazeTargetFrame& frame);
#endif

}  // namespace dds_bridge
}  // namespace visual_events
