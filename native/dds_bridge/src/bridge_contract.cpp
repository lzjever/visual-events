#include "visual_events/dds_bridge/bridge_contract.hpp"

namespace visual_events {
namespace dds_bridge {

std::string_view ProbeStatusJson() {
    return R"({"protocol_version":1,"type":"status","code":"probe_ok","message":"native DDS bridge probe ok","mode":"probe"})";
}

}  // namespace dds_bridge
}  // namespace visual_events
