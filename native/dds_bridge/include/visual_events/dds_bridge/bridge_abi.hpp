#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace visual_events {
namespace dds_bridge {

struct CameraJpegFrame {
    int64_t dds_timestamp_ns;
    int64_t received_monotonic_ns;
    std::string camera_name;
    int64_t width;
    int64_t height;
    std::string encoding;
    int64_t step;
    std::vector<uint8_t> data;
};

struct HeadStateFrame {
    int64_t dds_timestamp_ns;
    int64_t received_monotonic_ns;
    bool valid;
    std::string state;
    double yaw_rad;
    double pitch_rad;
    double yaw_vel_rad_s;
    double pitch_vel_rad_s;
};

struct GazeTargetFrame {
    int64_t schema_version;
    std::string camera;
    int64_t frame_id;
    int64_t frame_timestamp_ms;
    int64_t publish_timestamp_ms;
    bool valid;
    std::string state;
    int64_t target_track_id;
    double target_u;
    double target_v;
    double target_norm_x;
    double target_norm_y;
    int64_t image_width;
    int64_t image_height;
    double confidence;
    std::string reason;
    int64_t stale_after_ms;
};

struct GazeTargetParseResult {
    bool ok;
    GazeTargetFrame frame;
    std::string error;
};

int64_t MonotonicNowNs();

std::string EncodeStatusFrame(
    std::string_view code,
    std::string_view message,
    std::string_view mode = {});
std::string EncodeErrorFrame(std::string_view code, std::string_view message, bool fatal);
std::string EncodeCameraJpegFrame(const CameraJpegFrame& frame);
std::string EncodeHeadStateFrame(const HeadStateFrame& frame);

GazeTargetParseResult ParseGazeTargetLine(std::string_view line);

}  // namespace dds_bridge
}  // namespace visual_events
