#include "visual_events/dds_bridge/bridge_dds_types.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace visual_events {
namespace dds_bridge {
namespace {

#ifdef VISUAL_EVENTS_DDS_BRIDGE_FULL_BRIDGE
bool IsAllowedGazeTargetState(std::string_view state) {
    return state == "tracking" || state == "lost" || state == "stale" || state == "disabled";
}

HeadStateFrame InvalidHeadState(int64_t received_monotonic_ns) {
    return HeadStateFrame{
        0,
        received_monotonic_ns,
        false,
        "unknown",
        0.0,
        0.0,
        0.0,
        0.0,
    };
}

bool HeadNumericsFinite(const visual_events::msg::dds_::HeadStateV1_& frame) {
    return std::isfinite(frame.yaw_rad()) && std::isfinite(frame.pitch_rad()) &&
           std::isfinite(frame.yaw_vel_rad_s()) && std::isfinite(frame.pitch_vel_rad_s());
}

int64_t TimestampMsToNs(int64_t timestamp_ms) {
    constexpr int64_t kNsPerMs = 1000000;
    if (timestamp_ms > std::numeric_limits<int64_t>::max() / kNsPerMs ||
        timestamp_ms < std::numeric_limits<int64_t>::min() / kNsPerMs) {
        return 0;
    }
    return timestamp_ms * kNsPerMs;
}

uint32_t RequireUint32(int64_t value, std::string_view field) {
    if (value < 0 || value > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument(std::string(field) + " must fit uint32");
    }
    return static_cast<uint32_t>(value);
}

float RequireFiniteFloat(double value, std::string_view field) {
    if (!std::isfinite(value) ||
        value < -static_cast<double>(std::numeric_limits<float>::max()) ||
        value > static_cast<double>(std::numeric_limits<float>::max())) {
        throw std::invalid_argument(std::string(field) + " must be finite and fit float");
    }
    const float narrowed = static_cast<float>(value);
    if (!std::isfinite(narrowed)) {
        throw std::invalid_argument(std::string(field) + " must be finite and fit float");
    }
    return narrowed;
}
#endif

}  // namespace

CameraJpegFrame CameraFrameToAbi(
    const unitree_camera::msg::dds_::CameraFrame_& frame,
    int64_t received_monotonic_ns) {
    if (frame.timestamp_ns() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::invalid_argument("timestamp_ns must fit int64");
    }
    return CameraJpegFrame{
        static_cast<int64_t>(frame.timestamp_ns()),
        received_monotonic_ns,
        frame.camera_name(),
        static_cast<int64_t>(frame.width()),
        static_cast<int64_t>(frame.height()),
        frame.encoding(),
        static_cast<int64_t>(frame.step()),
        frame.data(),
    };
}

#ifdef VISUAL_EVENTS_DDS_BRIDGE_FULL_BRIDGE
HeadStateFrame HeadStateToAbi(
    const visual_events::msg::dds_::HeadStateV1_& frame,
    int64_t received_monotonic_ns,
    double stationary_yaw_vel_rad_s,
    double stationary_pitch_vel_rad_s) {
    if (frame.schema_version() != 1 || !frame.valid() || !HeadNumericsFinite(frame) ||
        !std::isfinite(stationary_yaw_vel_rad_s) ||
        !std::isfinite(stationary_pitch_vel_rad_s) ||
        stationary_yaw_vel_rad_s < 0.0 || stationary_pitch_vel_rad_s < 0.0) {
        return InvalidHeadState(received_monotonic_ns);
    }

    const bool stationary =
        std::fabs(frame.yaw_vel_rad_s()) <= stationary_yaw_vel_rad_s &&
        std::fabs(frame.pitch_vel_rad_s()) <= stationary_pitch_vel_rad_s;
    return HeadStateFrame{
        TimestampMsToNs(frame.timestamp_ms()),
        received_monotonic_ns,
        true,
        stationary ? "stationary" : "moving",
        frame.yaw_rad(),
        frame.pitch_rad(),
        frame.yaw_vel_rad_s(),
        frame.pitch_vel_rad_s(),
    };
}

visual_events::msg::dds_::GazeTargetV1_ GazeTargetFrameToDds(const GazeTargetFrame& frame) {
    if (frame.schema_version != 1) {
        throw std::invalid_argument("schema_version must be 1");
    }
    if (!IsAllowedGazeTargetState(frame.state)) {
        throw std::invalid_argument("state must be tracking, lost, stale, or disabled");
    }
    if (frame.valid != (frame.state == "tracking")) {
        throw std::invalid_argument("valid must be true only when state is tracking");
    }

    return visual_events::msg::dds_::GazeTargetV1_{
        RequireUint32(frame.schema_version, "schema_version"),
        frame.camera,
        frame.frame_id,
        frame.frame_timestamp_ms,
        frame.publish_timestamp_ms,
        frame.valid,
        frame.state,
        frame.target_track_id,
        RequireFiniteFloat(frame.target_u, "target_u"),
        RequireFiniteFloat(frame.target_v, "target_v"),
        RequireFiniteFloat(frame.target_norm_x, "target_norm_x"),
        RequireFiniteFloat(frame.target_norm_y, "target_norm_y"),
        RequireUint32(frame.image_width, "image_width"),
        RequireUint32(frame.image_height, "image_height"),
        RequireFiniteFloat(frame.confidence, "confidence"),
        frame.reason,
        RequireUint32(frame.stale_after_ms, "stale_after_ms"),
    };
}
#endif

}  // namespace dds_bridge
}  // namespace visual_events
