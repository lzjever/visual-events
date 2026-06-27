#include "visual_events/dds_bridge/bridge_abi.hpp"

#ifdef VISUAL_EVENTS_DDS_BRIDGE_FULL_BRIDGE
#include "visual_events/dds_bridge/runtime_loop.hpp"
#include "visual_events/dds_bridge/runtime_options.hpp"
#include "visual_events/dds_bridge/unitree_channel_runtime.hpp"
#endif

#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <string_view>

namespace {

bool IsProbeArg(const char* arg) {
    return std::strcmp(arg, "--probe") == 0;
}

int EmitFatal(std::string_view code, std::string_view message, int exit_code = 1) {
    std::cerr << message << '\n';
    std::cout << visual_events::dds_bridge::EncodeErrorFrame(code, message, true) << '\n';
    std::cout.flush();
    return exit_code;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && IsProbeArg(argv[1])) {
        std::cout << visual_events::dds_bridge::EncodeStatusFrame(
                         "probe_ok",
                         "native DDS bridge runtime probe ok",
                         "probe")
                  << '\n';
        return 0;
    }

    if (argc != 1) {
        return EmitFatal(
            "invalid_arguments",
            "usage: visual_events_dds_bridge [--probe]",
            2);
    }

#ifdef VISUAL_EVENTS_DDS_BRIDGE_FULL_BRIDGE
    const auto parsed = visual_events::dds_bridge::RuntimeOptionsFromEnv();
    if (!parsed.ok) {
        return EmitFatal("invalid_runtime_options", parsed.error);
    }

    try {
        std::unique_ptr<visual_events::dds_bridge::RuntimeBackend> backend =
            visual_events::dds_bridge::CreateUnitreeRuntimeBackend(parsed.options);
        return visual_events::dds_bridge::RunRuntimeLoop(
            *backend,
            std::cin,
            std::cout,
            std::cerr);
    } catch (const std::exception& exc) {
        return EmitFatal("dds_runtime_failed", exc.what());
    } catch (...) {
        return EmitFatal("dds_runtime_failed", "unknown DDS runtime failure");
    }
#else
    std::cerr << "DDS runtime backend is not implemented in this skeleton build\n";
    std::cout << visual_events::dds_bridge::EncodeErrorFrame(
                     "dds_runtime_not_implemented",
                     "native DDS runtime backend is not implemented",
                     true)
              << '\n';
    return 1;
#endif
}
