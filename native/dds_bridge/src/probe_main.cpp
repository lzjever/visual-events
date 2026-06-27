#include "visual_events/dds_bridge/bridge_contract.hpp"

#include "unitree_camera/msg/dds/CameraFrame_.hpp"

#include <cstring>
#include <iostream>

namespace cdr = org::eclipse::cyclonedds::core::cdr;

namespace {

bool IsProbeArg(const char* arg) {
    return std::strcmp(arg, "--probe") == 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 2 || (argc == 2 && !IsProbeArg(argv[1]))) {
        std::cerr << "usage: visual_events_dds_bridge_probe [--probe]\n";
        return 2;
    }

    auto& camera_props =
        cdr::get_type_props<unitree_camera::msg::dds_::CameraFrame_>();
    if (camera_props.empty()) {
        std::cerr << "CameraFrame_ type properties are unavailable\n";
        return 1;
    }

    std::cerr << "visual_events_dds_bridge_probe: checked Unitree SDK2 link and CameraFrame_ type properties\n";
    std::cout << visual_events::dds_bridge::ProbeStatusJson() << '\n';
    return 0;
}
