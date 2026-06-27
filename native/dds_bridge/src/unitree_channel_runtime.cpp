#include "visual_events/dds_bridge/unitree_channel_runtime.hpp"

#ifndef VISUAL_EVENTS_DDS_BRIDGE_FULL_BRIDGE
#error "unitree_channel_runtime requires VISUAL_EVENTS_DDS_BRIDGE_FULL_BRIDGE"
#endif

#include "visual_events/dds_bridge/bridge_dds_types.hpp"

#include "gaze_target_v1.hpp"
#include "head_state_v1.hpp"
#include "unitree/robot/channel/channel_factory.hpp"
#include "unitree/robot/channel/channel_publisher.hpp"
#include "unitree/robot/channel/channel_subscriber.hpp"
#include "unitree_camera/msg/dds/CameraFrame_.hpp"

#include <atomic>
#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

namespace visual_events {
namespace dds_bridge {
namespace {

class UnitreeChannelSession final : public RuntimeBackend {
public:
    explicit UnitreeChannelSession(const RuntimeOptions& options) : options_(options) {}

    UnitreeChannelSession(const UnitreeChannelSession&) = delete;
    UnitreeChannelSession& operator=(const UnitreeChannelSession&) = delete;

    ~UnitreeChannelSession() {
        Close();
    }

    void ConstructChannels() {
        unitree::robot::ChannelFactory::Instance()->Init(options_.domain, options_.network);
        factory_inited_ = true;
        camera_subscriber_.InitChannel([this](const void* sample) { OnCameraFrame(sample); });
        head_subscriber_.InitChannel([this](const void* sample) { OnHeadState(sample); });
        gaze_publisher_.InitChannel();
        channels_open_ = true;
    }

    bool Start(const RuntimeBackendCallbacks& callbacks, std::string* error) override {
        {
            std::lock_guard<std::mutex> lock(callbacks_mutex_);
            callbacks_ = callbacks;
        }
        try {
            ConstructChannels();
            return true;
        } catch (const std::exception& exc) {
            if (error != nullptr) {
                *error = exc.what();
            }
        } catch (...) {
            if (error != nullptr) {
                *error = "unknown Unitree channel startup error";
            }
        }
        return false;
    }

    bool PublishGaze(const GazeTargetFrame& frame, std::string* error) override {
        try {
            const auto msg = GazeTargetFrameToDds(frame);
            if (!gaze_publisher_.Write(msg, 0)) {
                if (error != nullptr) {
                    *error = "Unitree gaze_target Write returned false";
                }
                return false;
            }
            return true;
        } catch (const std::exception& exc) {
            if (error != nullptr) {
                *error = exc.what();
            }
        } catch (...) {
            if (error != nullptr) {
                *error = "unknown Unitree gaze_target publish error";
            }
        }
        return false;
    }

    void Close() override {
        if (channels_open_) {
            head_subscriber_.CloseChannel();
            camera_subscriber_.CloseChannel();
            gaze_publisher_.CloseChannel();
            channels_open_ = false;
        }
        {
            std::lock_guard<std::mutex> lock(callbacks_mutex_);
            callbacks_ = RuntimeBackendCallbacks{};
        }
        if (factory_inited_) {
            unitree::robot::ChannelFactory::Instance()->Release();
            factory_inited_ = false;
        }
    }

private:
    void OnCameraFrame(const void* sample) {
        ++camera_callbacks_;
        RuntimeBackendCallbacks callbacks = CopyCallbacks();
        if (sample == nullptr || !callbacks.camera) {
            return;
        }
        try {
            const auto* frame =
                static_cast<const unitree_camera::msg::dds_::CameraFrame_*>(sample);
            callbacks.camera(CameraFrameToAbi(*frame, MonotonicNowNs()));
        } catch (const std::exception& exc) {
            EmitFatal("camera_frame_mapping_failed", exc.what());
        } catch (...) {
            EmitFatal("camera_frame_mapping_failed", "unknown camera frame mapping error");
        }
    }

    void OnHeadState(const void* sample) {
        ++head_callbacks_;
        RuntimeBackendCallbacks callbacks = CopyCallbacks();
        if (sample == nullptr || !callbacks.head) {
            return;
        }
        try {
            const auto* frame = static_cast<const visual_events::msg::dds_::HeadStateV1_*>(sample);
            callbacks.head(HeadStateToAbi(*frame, MonotonicNowNs()));
        } catch (const std::exception& exc) {
            EmitFatal("head_state_mapping_failed", exc.what());
        } catch (...) {
            EmitFatal("head_state_mapping_failed", "unknown head_state mapping error");
        }
    }

    void EmitFatal(std::string code, std::string message) {
        RuntimeBackendCallbacks callbacks = CopyCallbacks();
        if (callbacks.fatal) {
            callbacks.fatal(std::move(code), std::move(message));
        }
    }

    RuntimeBackendCallbacks CopyCallbacks() {
        std::lock_guard<std::mutex> lock(callbacks_mutex_);
        return callbacks_;
    }

    const RuntimeOptions& options_;
    bool factory_inited_ = false;
    bool channels_open_ = false;
    std::mutex callbacks_mutex_;
    RuntimeBackendCallbacks callbacks_;
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

std::unique_ptr<RuntimeBackend> CreateUnitreeRuntimeBackend(const RuntimeOptions& options) {
    return std::make_unique<UnitreeChannelSession>(options);
}

}  // namespace dds_bridge
}  // namespace visual_events
