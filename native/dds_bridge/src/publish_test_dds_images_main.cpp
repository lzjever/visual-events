#include "visual_events/dds_bridge/pc_test_tools.hpp"
#include "visual_events/dds_bridge/runtime_options.hpp"

#include "unitree/robot/channel/channel_publisher.hpp"
#include "unitree_camera/msg/dds/CameraFrame_.hpp"

#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string input;
    int64_t count = 0;
    double hz = 0.0;
    std::string camera_name = "front";
};

constexpr const char* kUsage =
    "usage: visual_events_dds_bridge_publish_test_dds_images "
    "--input <path> --count <n> --hz <hz> [--camera-name <name>]";

bool NeedValue(int index, int argc, const char* flag, std::string* error) {
    if (index + 1 < argc) {
        return true;
    }
    *error = std::string(flag) + " requires a value";
    return false;
}

bool ParseArgs(int argc, char** argv, Args* args, std::string* error) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--input") == 0) {
            if (!NeedValue(i, argc, argv[i], error)) {
                return false;
            }
            args->input = argv[++i];
        } else if (std::strcmp(argv[i], "--count") == 0) {
            if (!NeedValue(i, argc, argv[i], error)) {
                return false;
            }
            if (!visual_events::dds_bridge::pc_test_tools::ParsePositiveInt64(
                    argv[++i],
                    "--count",
                    &args->count,
                    error)) {
                return false;
            }
        } else if (std::strcmp(argv[i], "--hz") == 0) {
            if (!NeedValue(i, argc, argv[i], error)) {
                return false;
            }
            if (!visual_events::dds_bridge::pc_test_tools::ParsePositiveDouble(
                    argv[++i],
                    "--hz",
                    &args->hz,
                    error)) {
                return false;
            }
        } else if (std::strcmp(argv[i], "--camera-name") == 0) {
            if (!NeedValue(i, argc, argv[i], error)) {
                return false;
            }
            args->camera_name = argv[++i];
            if (args->camera_name.empty()) {
                *error = "--camera-name must be non-empty";
                return false;
            }
        } else {
            *error = std::string("unknown argument: ") + argv[i];
            return false;
        }
    }

    if (args->input.empty() || args->count <= 0 || args->hz <= 0.0) {
        *error = kUsage;
        return false;
    }
    return true;
}

std::string EncodeSummary(
    const visual_events::dds_bridge::RuntimeOptions& options,
    const Args& args,
    const visual_events::dds_bridge::pc_test_tools::JpegDimensions& last_dimensions,
    int64_t published,
    size_t input_count,
    size_t last_payload_size) {
    namespace tools = visual_events::dds_bridge::pc_test_tools;
    std::ostringstream out;
    out << "{\"protocol_version\":1"
        << ",\"type\":\"status\""
        << ",\"code\":\"publish_test_dds_images_ok\""
        << ",\"message\":\"published DDS camera image test frames\""
        << ",\"published\":" << published
        << ",\"input_count\":" << input_count
        << ",\"camera_name\":" << tools::JsonEscape(args.camera_name)
        << ",\"camera_topic\":" << tools::JsonEscape(options.camera_topic)
        << ",\"width\":" << last_dimensions.width
        << ",\"height\":" << last_dimensions.height
        << ",\"encoding\":\"JPEG\""
        << ",\"payload_size_bytes\":" << last_payload_size
        << "}";
    return out.str();
}

}  // namespace

int main(int argc, char** argv) {
    namespace tools = visual_events::dds_bridge::pc_test_tools;

    Args args;
    std::string error;
    if (!ParseArgs(argc, argv, &args, &error)) {
        if (error != kUsage) {
            error = error + "; " + kUsage;
        }
        return tools::EmitFatal("invalid_arguments", error, 2);
    }

    const auto parsed_options = visual_events::dds_bridge::RuntimeOptionsFromEnv();
    if (!parsed_options.ok) {
        return tools::EmitFatal("invalid_runtime_options", parsed_options.error);
    }

    try {
        const std::vector<std::filesystem::path> inputs = tools::ExpandJpegInput(args.input);
        tools::UnitreeFactoryScope factory(parsed_options.options);
        unitree::robot::ChannelPublisher<unitree_camera::msg::dds_::CameraFrame_> publisher(
            parsed_options.options.camera_topic);
        publisher.InitChannel();

        tools::JpegDimensions last_dimensions;
        size_t last_payload_size = 0;
        for (int64_t i = 0; i < args.count; ++i) {
            const auto& path = inputs[static_cast<size_t>(i) % inputs.size()];
            std::vector<uint8_t> bytes = tools::ReadFileBytes(path);
            const auto dimensions = tools::ParseJpegDimensions(bytes);
            if (bytes.size() > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("JPEG payload is too large for CameraFrame_.step");
            }

            unitree_camera::msg::dds_::CameraFrame_ frame;
            frame.timestamp_ns(tools::MonotonicNowUint64Ns());
            frame.camera_name(args.camera_name);
            frame.width(dimensions.width);
            frame.height(dimensions.height);
            frame.encoding("JPEG");
            frame.step(static_cast<uint32_t>(bytes.size()));
            frame.data(std::move(bytes));

            if (!publisher.Write(frame, 0)) {
                throw std::runtime_error("Unitree camera Write returned false");
            }
            last_dimensions = dimensions;
            last_payload_size = frame.data().size();
            if (i + 1 < args.count) {
                tools::SleepForHz(args.hz);
            }
        }

        publisher.CloseChannel();
        std::cout << EncodeSummary(
                         parsed_options.options,
                         args,
                         last_dimensions,
                         args.count,
                         inputs.size(),
                         last_payload_size)
                  << '\n';
        return 0;
    } catch (const std::exception& exc) {
        return tools::EmitFatal("publish_test_dds_images_failed", exc.what());
    } catch (...) {
        return tools::EmitFatal("publish_test_dds_images_failed", "unknown image publish failure");
    }
}
