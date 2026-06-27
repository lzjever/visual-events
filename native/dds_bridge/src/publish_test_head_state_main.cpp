#include "visual_events/dds_bridge/bridge_dds_types.hpp"
#include "visual_events/dds_bridge/pc_test_tools.hpp"
#include "visual_events/dds_bridge/runtime_options.hpp"

#include "head_state_v1.hpp"
#include "unitree/robot/channel/channel_publisher.hpp"

#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

struct Args {
    std::string state;
    int64_t count = 0;
    double hz = 0.0;
};

constexpr const char* kUsage =
    "usage: visual_events_dds_bridge_publish_test_head_state "
    "--state stationary|moving|unknown --count <n> --hz <hz>";

bool NeedValue(int index, int argc, const char* flag, std::string* error) {
    if (index + 1 < argc) {
        return true;
    }
    *error = std::string(flag) + " requires a value";
    return false;
}

bool IsAllowedState(const std::string& state) {
    return state == "stationary" || state == "moving" || state == "unknown";
}

bool ParseArgs(int argc, char** argv, Args* args, std::string* error) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--state") == 0) {
            if (!NeedValue(i, argc, argv[i], error)) {
                return false;
            }
            args->state = argv[++i];
            if (!IsAllowedState(args->state)) {
                *error = "--state must be stationary|moving|unknown";
                return false;
            }
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
        } else {
            *error = std::string("unknown argument: ") + argv[i];
            return false;
        }
    }

    if (args->state.empty() || args->count <= 0 || args->hz <= 0.0) {
        *error = kUsage;
        return false;
    }
    return true;
}

visual_events::msg::dds_::HeadStateV1_ MakeHeadStateMessage(const std::string& state) {
    const int64_t timestamp_ms = visual_events::dds_bridge::pc_test_tools::SystemNowMs();
    if (state == "stationary") {
        return visual_events::msg::dds_::HeadStateV1_{
            1,
            timestamp_ms,
            true,
            0.0,
            0.0,
            0.0,
            0.0,
        };
    }
    if (state == "moving") {
        return visual_events::msg::dds_::HeadStateV1_{
            1,
            timestamp_ms,
            true,
            0.0,
            0.0,
            0.08,
            0.0,
        };
    }
    return visual_events::msg::dds_::HeadStateV1_{
        1,
        timestamp_ms,
        false,
        0.0,
        0.0,
        0.0,
        0.0,
    };
}

std::string EncodeSummary(
    const visual_events::dds_bridge::RuntimeOptions& options,
    const Args& args,
    const visual_events::msg::dds_::HeadStateV1_& last_message,
    const visual_events::dds_bridge::HeadStateFrame& mapped) {
    namespace tools = visual_events::dds_bridge::pc_test_tools;
    std::ostringstream out;
    out << "{\"protocol_version\":1"
        << ",\"type\":\"status\""
        << ",\"code\":\"publish_test_head_state_ok\""
        << ",\"message\":\"published DDS head_state test frames\""
        << ",\"published\":" << args.count
        << ",\"state\":" << tools::JsonEscape(args.state)
        << ",\"head_state_topic\":" << tools::JsonEscape(options.head_state_topic)
        << ",\"dds_valid\":" << (last_message.valid() ? "true" : "false")
        << ",\"mapped_valid\":" << (mapped.valid ? "true" : "false")
        << ",\"mapped_state\":" << tools::JsonEscape(mapped.state)
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
        tools::UnitreeFactoryScope factory(parsed_options.options);
        unitree::robot::ChannelPublisher<visual_events::msg::dds_::HeadStateV1_> publisher(
            parsed_options.options.head_state_topic);
        publisher.InitChannel();

        visual_events::msg::dds_::HeadStateV1_ last_message;
        for (int64_t i = 0; i < args.count; ++i) {
            last_message = MakeHeadStateMessage(args.state);
            if (!publisher.Write(last_message, 0)) {
                throw std::runtime_error("Unitree head_state Write returned false");
            }
            if (i + 1 < args.count) {
                tools::SleepForHz(args.hz);
            }
        }

        const auto mapped = visual_events::dds_bridge::HeadStateToAbi(
            last_message,
            visual_events::dds_bridge::MonotonicNowNs());
        publisher.CloseChannel();
        std::cout << EncodeSummary(parsed_options.options, args, last_message, mapped) << '\n';
        return 0;
    } catch (const std::exception& exc) {
        return tools::EmitFatal("publish_test_head_state_failed", exc.what());
    } catch (...) {
        return tools::EmitFatal("publish_test_head_state_failed", "unknown head_state publish failure");
    }
}
