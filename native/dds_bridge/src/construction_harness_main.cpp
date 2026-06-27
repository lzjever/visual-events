#include "visual_events/dds_bridge/bridge_abi.hpp"
#include "visual_events/dds_bridge/runtime_options.hpp"
#include "visual_events/dds_bridge/unitree_channel_runtime.hpp"

#include <cstring>
#include <exception>
#include <iostream>
#include <string_view>

namespace {

bool IsArg(const char* arg, const char* expected) {
    return std::strcmp(arg, expected) == 0;
}

int EmitFatal(std::string_view code, std::string_view message) {
    std::cerr << message << '\n';
    std::cout << visual_events::dds_bridge::EncodeErrorFrame(code, message, true) << '\n';
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2 ||
        (!IsArg(argv[1], "--print-options") && !IsArg(argv[1], "--construct-once"))) {
        return EmitFatal(
            "invalid_arguments",
            "usage: visual_events_dds_bridge_construction_harness "
            "[--print-options|--construct-once]");
    }

    const auto parsed = visual_events::dds_bridge::RuntimeOptionsFromEnv();
    if (!parsed.ok) {
        return EmitFatal("invalid_runtime_options", parsed.error);
    }

    if (IsArg(argv[1], "--print-options")) {
        std::cout << visual_events::dds_bridge::EncodeRuntimeOptionsStatus(
                         parsed.options,
                         "print_options")
                  << '\n';
        return 0;
    }

    try {
        visual_events::dds_bridge::ConstructUnitreeChannelsOnce(parsed.options);
    } catch (const std::exception& exc) {
        return EmitFatal("unitree_channel_construction_failed", exc.what());
    } catch (...) {
        return EmitFatal("unitree_channel_construction_failed", "unknown construction error");
    }

    std::cout << visual_events::dds_bridge::EncodeStatusFrame(
                     "construction_ok",
                     "native Unitree channel construction ok",
                     "construct_once")
              << '\n';
    return 0;
}
