#include "visual_events/dds_bridge/unitree_channel_runtime.hpp"

#ifndef VISUAL_EVENTS_DDS_BRIDGE_FULL_BRIDGE
#error "unitree_channel_runtime requires VISUAL_EVENTS_DDS_BRIDGE_FULL_BRIDGE"
#endif

#include "gaze_target_v1.hpp"
#include "head_state_v1.hpp"
#include "unitree/robot/channel/channel_factory.hpp"
#include "unitree/robot/channel/channel_publisher.hpp"
#include "unitree/robot/channel/channel_subscriber.hpp"
#include "unitree_camera/msg/dds/CameraFrame_.hpp"

#include <atomic>
#include <cstddef>

namespace visual_events {
namespace dds_bridge {
namespace {

class UnitreeChannelSession {
public:
    explicit UnitreeChannelSession(const RuntimeOptions& options) : options_(options) {}

    UnitreeChannelSession(const UnitreeChannelSession&) = delete;
    UnitreeChannelSession& operator=(const UnitreeChannelSession&) = delete;

    ~UnitreeChannelSession() {
        gaze_publisher_.CloseChannel();
        head_subscriber_.CloseChannel();
        camera_subscriber_.CloseChannel();
        if (factory_inited_) {
            unitree::robot::ChannelFactory::Instance()->Release();
        }
    }

    void ConstructChannels() {
        unitree::robot::ChannelFactory::Instance()->Init(options_.domain, options_.network);
        factory_inited_ = true;
        camera_subscriber_.InitChannel([this](const void* sample) { OnCameraFrame(sample); });
        head_subscriber_.InitChannel([this](const void* sample) { OnHeadState(sample); });
        gaze_publisher_.InitChannel();
    }

private:
    void OnCameraFrame(const void*) {
        ++camera_callbacks_;
    }

    void OnHeadState(const void*) {
        ++head_callbacks_;
    }

    const RuntimeOptions& options_;
    bool factory_inited_ = false;
    std::atomic<size_t> camera_callbacks_{0};
    std::atomic<size_t> head_callbacks_{0};
    unitree::robot::ChannelSubscriber<unitree_camera::msg::dds_::CameraFrame_> camera_subscriber_{
        options_.camera_topic};
    unitree::robot::ChannelSubscriber<visual_events::msg::dds_::HeadStateV1_> head_subscriber_{
        options_.head_state_topic};
    unitree::robot::ChannelPublisher<visual_events::msg::dds_::GazeTargetV1_> gaze_publisher_{
        options_.gaze_topic};
};

}  // namespace

void ConstructUnitreeChannelsOnce(const RuntimeOptions& options) {
    UnitreeChannelSession session(options);
    session.ConstructChannels();
}

}  // namespace dds_bridge
}  // namespace visual_events
