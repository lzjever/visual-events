#include "visual_events/dds_bridge/runtime_loop.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

int AsyncFatalDelayMsFromEnv() {
    const char* raw = std::getenv("VISUAL_EVENTS_RUNTIME_LOOP_HARNESS_ASYNC_FATAL_MS");
    if (raw == nullptr || raw[0] == '\0') {
        return 0;
    }
    try {
        const int delay_ms = std::stoi(raw);
        return delay_ms > 0 ? delay_ms : 0;
    } catch (...) {
        return 0;
    }
}

class FakeRuntimeBackend final : public visual_events::dds_bridge::RuntimeBackend {
public:
    ~FakeRuntimeBackend() override {
        Close();
    }

    bool Start(
        const visual_events::dds_bridge::RuntimeBackendCallbacks& callbacks,
        std::string*) override {
        callbacks_ = callbacks;
        if (callbacks_.camera) {
            callbacks_.camera(FakeCameraFrame());
        }
        if (callbacks_.head) {
            callbacks_.head(FakeHeadStateFrame());
        }
        const int async_fatal_delay_ms = AsyncFatalDelayMsFromEnv();
        if (async_fatal_delay_ms > 0 && callbacks_.fatal) {
            async_fatal_thread_ = std::thread([this, async_fatal_delay_ms] {
                std::this_thread::sleep_for(std::chrono::milliseconds(async_fatal_delay_ms));
                if (callbacks_.fatal) {
                    callbacks_.fatal(
                        "async_backend_fatal",
                        "fake backend fatal after start");
                }
            });
        }
        return true;
    }

    bool PublishGaze(const visual_events::dds_bridge::GazeTargetFrame& frame, std::string*) override {
        ++published_gaze_targets_;
        last_frame_id_ = frame.frame_id;
        return true;
    }

    void Close() override {
        if (async_fatal_thread_.joinable()) {
            async_fatal_thread_.join();
        }
        closed_ = true;
    }

    int published_gaze_targets() const {
        return published_gaze_targets_;
    }

    int64_t last_frame_id() const {
        return last_frame_id_;
    }

    bool closed() const {
        return closed_;
    }

private:
    static visual_events::dds_bridge::CameraJpegFrame FakeCameraFrame() {
        return visual_events::dds_bridge::CameraJpegFrame{
            123456789,
            987654321,
            "front",
            2,
            2,
            "JPEG",
            4,
            std::vector<uint8_t>{0xFF, 0xD8, 0xFF, 0xD9},
        };
    }

    static visual_events::dds_bridge::HeadStateFrame FakeHeadStateFrame() {
        return visual_events::dds_bridge::HeadStateFrame{
            123456790,
            987654322,
            true,
            "stationary",
            0.1,
            -0.2,
            0.01,
            0.02,
        };
    }

    visual_events::dds_bridge::RuntimeBackendCallbacks callbacks_;
    std::thread async_fatal_thread_;
    int published_gaze_targets_ = 0;
    int64_t last_frame_id_ = -1;
    bool closed_ = false;
};

}  // namespace

int main(int argc, char**) {
    if (argc != 1) {
        std::cerr << "usage: visual_events_dds_bridge_runtime_loop_harness\n";
        std::cout << visual_events::dds_bridge::EncodeErrorFrame(
                         "invalid_arguments",
                         "usage: visual_events_dds_bridge_runtime_loop_harness",
                         true)
                  << '\n';
        return 2;
    }

    FakeRuntimeBackend backend;
    const int rc = visual_events::dds_bridge::RunRuntimeLoop(
        backend,
        std::cin,
        std::cout,
        std::cerr);
    std::cerr << "published_gaze_target=" << backend.published_gaze_targets()
              << " last_frame_id=" << backend.last_frame_id()
              << " closed=" << (backend.closed() ? "true" : "false") << '\n';
    return rc;
}
