#include "visual_events/dds_bridge/bridge_abi.hpp"

#include <cstring>
#include <iostream>

namespace {

bool IsProbeArg(const char* arg) {
    return std::strcmp(arg, "--probe") == 0;
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
        std::cerr << "usage: visual_events_dds_bridge [--probe]\n";
        std::cout << visual_events::dds_bridge::EncodeErrorFrame(
                         "invalid_arguments",
                         "usage: visual_events_dds_bridge [--probe]",
                         true)
                  << '\n';
        return 2;
    }

    std::cerr << "DDS runtime backend is not implemented in this skeleton build\n";
    std::cout << visual_events::dds_bridge::EncodeErrorFrame(
                     "dds_runtime_not_implemented",
                     "native DDS runtime backend is not implemented",
                     true)
              << '\n';
    return 1;
}
