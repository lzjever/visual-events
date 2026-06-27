#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace visual_events {
namespace dds_bridge {

struct TopicContract {
    std::string_view name;
    std::string_view type_name;
    uint32_t deadline_ms;
    uint32_t lifespan_ms;
    uint32_t liveliness_lease_ms;
};

inline constexpr uint32_t kProtocolVersion = 1;

inline constexpr TopicContract kCameraTopic{
    "/camera/image/jpeg",
    "unitree_camera::msg::dds_::CameraFrame_",
    150,
    300,
    1000,
};

inline constexpr TopicContract kHeadTopic{
    "/robot/head_state",
    "visual_events::msg::dds_::HeadStateV1_",
    150,
    250,
    500,
};

inline constexpr TopicContract kGazeTopic{
    "/visual_events/gaze_target",
    "visual_events::msg::dds_::GazeTargetV1_",
    150,
    250,
    500,
};

inline constexpr std::array<TopicContract, 3> kAllowedTopics{
    kCameraTopic,
    kHeadTopic,
    kGazeTopic,
};

std::string_view ProbeStatusJson();

}  // namespace dds_bridge
}  // namespace visual_events
