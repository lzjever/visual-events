#include "visual_events/dds_bridge/bridge_abi.hpp"

#include <iostream>
#include <string>
#include <vector>

namespace {

visual_events::dds_bridge::CameraJpegFrame FakeCameraFrame() {
    const int64_t now_ns = visual_events::dds_bridge::MonotonicNowNs();
    return visual_events::dds_bridge::CameraJpegFrame{
        now_ns,
        now_ns,
        "front",
        2,
        2,
        "JPEG",
        4,
        std::vector<uint8_t>{0xFF, 0xD8, 0xFF, 0xD9},
    };
}

visual_events::dds_bridge::HeadStateFrame FakeHeadStateFrame() {
    const int64_t now_ns = visual_events::dds_bridge::MonotonicNowNs();
    return visual_events::dds_bridge::HeadStateFrame{
        now_ns,
        now_ns,
        true,
        "stationary",
        0.0,
        0.0,
        0.0,
        0.0,
    };
}

}  // namespace

int main(int argc, char**) {
    if (argc != 1) {
        std::cerr << "usage: visual_events_dds_bridge_abi_harness\n";
        std::cout << visual_events::dds_bridge::EncodeErrorFrame(
                         "invalid_arguments",
                         "usage: visual_events_dds_bridge_abi_harness",
                         true)
                  << '\n';
        return 2;
    }

    std::cout << visual_events::dds_bridge::EncodeCameraJpegFrame(FakeCameraFrame()) << '\n';
    std::cout << visual_events::dds_bridge::EncodeHeadStateFrame(FakeHeadStateFrame()) << '\n';

    std::string line;
    int accepted = 0;
    while (std::getline(std::cin, line)) {
        const auto parsed = visual_events::dds_bridge::ParseGazeTargetLine(line);
        if (!parsed.ok) {
            std::cerr << "invalid gaze_target: " << parsed.error << '\n';
            std::cout << visual_events::dds_bridge::EncodeErrorFrame(
                             "invalid_gaze_target",
                             parsed.error,
                             true)
                      << '\n';
            return 1;
        }
        ++accepted;
    }

    std::cerr << "accepted_gaze_target=" << accepted << '\n';
    return 0;
}
