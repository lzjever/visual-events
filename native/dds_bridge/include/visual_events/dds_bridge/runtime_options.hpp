#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace visual_events {
namespace dds_bridge {

struct RuntimeOptions {
    int32_t domain = 0;
    std::string network;
    std::string camera_topic;
    std::string head_state_topic;
    std::string gaze_topic;
};

struct RuntimeOptionsResult {
    bool ok = false;
    RuntimeOptions options;
    std::string error;
};

RuntimeOptionsResult RuntimeOptionsFromEnv();
std::string EncodeRuntimeOptionsStatus(const RuntimeOptions& options, std::string_view mode);

}  // namespace dds_bridge
}  // namespace visual_events
