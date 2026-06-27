#include "visual_events/dds_bridge/bridge_contract.hpp"

#ifdef VISUAL_EVENTS_DDS_BRIDGE_FULL_BRIDGE
#include "gaze_target_v1.hpp"
#include "head_state_v1.hpp"
#endif

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

#ifdef VISUAL_EVENTS_DDS_BRIDGE_FULL_BRIDGE
    auto& head_props =
        cdr::get_type_props<visual_events::msg::dds_::HeadStateV1_>();
    if (head_props.empty()) {
        std::cerr << "HeadStateV1_ type properties are unavailable\n";
        return 1;
    }

    auto& gaze_props =
        cdr::get_type_props<visual_events::msg::dds_::GazeTargetV1_>();
    if (gaze_props.empty()) {
        std::cerr << "GazeTargetV1_ type properties are unavailable\n";
        return 1;
    }

    std::cout << R"({"protocol_version":1,"type":"status","code":"probe_ok","message":"native DDS bridge probe ok","mode":"probe","type_support":["CameraFrame_","HeadStateV1_","GazeTargetV1_"]})"
              << '\n';
#else
    std::cout << visual_events::dds_bridge::ProbeStatusJson() << '\n';
#endif
    return 0;
}
