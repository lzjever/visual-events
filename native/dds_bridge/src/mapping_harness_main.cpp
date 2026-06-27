#include "visual_events/dds_bridge/bridge_abi.hpp"
#include "visual_events/dds_bridge/bridge_dds_types.hpp"

#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr int64_t kReceivedNs = 987654321;

std::string JsonEscape(std::string_view value) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char c : value) {
        switch (c) {
            case '"':
                out << "\\\"";
                break;
            case '\\':
                out << "\\\\";
                break;
            case '\b':
                out << "\\b";
                break;
            case '\f':
                out << "\\f";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                if (c < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(c) << std::dec;
                } else {
                    out << static_cast<char>(c);
                }
                break;
        }
    }
    out << '"';
    return out.str();
}

std::string EncodeFloat(float value) {
    std::ostringstream out;
    out << std::setprecision(9) << value;
    return out.str();
}

unitree_camera::msg::dds_::CameraFrame_ SampleCameraFrame() {
    unitree_camera::msg::dds_::CameraFrame_ frame;
    frame.timestamp_ns(123456789);
    frame.camera_name("front");
    frame.width(1280);
    frame.height(720);
    frame.encoding("jpeg");
    frame.step(4096);
    frame.data(std::vector<uint8_t>{0xFF, 0xD8, 0xFF, 0xD9});
    return frame;
}

visual_events::msg::dds_::HeadStateV1_ StationaryHeadState() {
    return visual_events::msg::dds_::HeadStateV1_{
        1,
        1710000000000,
        true,
        0.1,
        -0.2,
        0.01,
        -0.02,
    };
}

visual_events::msg::dds_::HeadStateV1_ MovingHeadState() {
    return visual_events::msg::dds_::HeadStateV1_{
        1,
        1710000000100,
        true,
        0.1,
        -0.2,
        0.031,
        0.0,
    };
}

visual_events::msg::dds_::HeadStateV1_ InvalidHeadState() {
    return visual_events::msg::dds_::HeadStateV1_{
        1,
        1710000000200,
        true,
        std::numeric_limits<double>::quiet_NaN(),
        0.0,
        0.0,
        0.0,
    };
}

std::string EncodeConstructedGaze(const visual_events::msg::dds_::GazeTargetV1_& frame) {
    std::ostringstream out;
    out << "{\"protocol_version\":1"
        << ",\"type\":\"gaze_target_constructed\""
        << ",\"schema_version\":" << frame.schema_version()
        << ",\"camera\":" << JsonEscape(frame.camera())
        << ",\"frame_id\":" << frame.frame_id()
        << ",\"frame_timestamp_ms\":" << frame.frame_timestamp_ms()
        << ",\"publish_timestamp_ms\":" << frame.publish_timestamp_ms()
        << ",\"valid\":" << (frame.valid() ? "true" : "false")
        << ",\"state\":" << JsonEscape(frame.state())
        << ",\"target_track_id\":" << frame.target_track_id()
        << ",\"target_u\":" << EncodeFloat(frame.target_u())
        << ",\"target_v\":" << EncodeFloat(frame.target_v())
        << ",\"target_norm_x\":" << EncodeFloat(frame.target_norm_x())
        << ",\"target_norm_y\":" << EncodeFloat(frame.target_norm_y())
        << ",\"image_width\":" << frame.image_width()
        << ",\"image_height\":" << frame.image_height()
        << ",\"confidence\":" << EncodeFloat(frame.confidence())
        << ",\"reason\":" << JsonEscape(frame.reason())
        << ",\"stale_after_ms\":" << frame.stale_after_ms()
        << "}";
    return out.str();
}

bool IsHelpArg(const char* arg) {
    return std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1) {
        if (argc == 2 && IsHelpArg(argv[1])) {
            std::cerr << "usage: visual_events_dds_bridge_mapping_harness\n";
            return 0;
        }
        std::cerr << "usage: visual_events_dds_bridge_mapping_harness\n";
        std::cout << visual_events::dds_bridge::EncodeErrorFrame(
                         "invalid_arguments",
                         "usage: visual_events_dds_bridge_mapping_harness",
                         true)
                  << '\n';
        return 2;
    }

    try {
        const auto camera = visual_events::dds_bridge::CameraFrameToAbi(
            SampleCameraFrame(),
            kReceivedNs);
        std::cout << visual_events::dds_bridge::EncodeCameraJpegFrame(camera) << '\n';
        std::cout << visual_events::dds_bridge::EncodeHeadStateFrame(
                         visual_events::dds_bridge::HeadStateToAbi(
                             StationaryHeadState(),
                             kReceivedNs))
                  << '\n';
        std::cout << visual_events::dds_bridge::EncodeHeadStateFrame(
                         visual_events::dds_bridge::HeadStateToAbi(MovingHeadState(), kReceivedNs))
                  << '\n';
        std::cout << visual_events::dds_bridge::EncodeHeadStateFrame(
                         visual_events::dds_bridge::HeadStateToAbi(InvalidHeadState(), kReceivedNs))
                  << '\n';

        std::string line;
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
            try {
                const auto dds = visual_events::dds_bridge::GazeTargetFrameToDds(parsed.frame);
                std::cout << EncodeConstructedGaze(dds) << '\n';
            } catch (const std::invalid_argument& exc) {
                std::cerr << "invalid gaze_target: " << exc.what() << '\n';
                std::cout << visual_events::dds_bridge::EncodeErrorFrame(
                                 "invalid_gaze_target",
                                 exc.what(),
                                 true)
                          << '\n';
                return 1;
            }
        }
    } catch (const std::exception& exc) {
        std::cerr << "mapping harness failed: " << exc.what() << '\n';
        std::cout << visual_events::dds_bridge::EncodeErrorFrame(
                         "mapping_harness_failed",
                         exc.what(),
                         true)
                  << '\n';
        return 1;
    }

    return 0;
}
