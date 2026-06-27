#pragma once

#include "visual_events/dds_bridge/bridge_abi.hpp"

#include <functional>
#include <iosfwd>
#include <string>

namespace visual_events {
namespace dds_bridge {

struct RuntimeBackendCallbacks {
    std::function<void(const CameraJpegFrame&)> camera;
    std::function<void(const HeadStateFrame&)> head;
    std::function<void(std::string code, std::string message)> fatal;
};

class RuntimeBackend {
public:
    virtual ~RuntimeBackend() = default;

    virtual bool Start(const RuntimeBackendCallbacks& callbacks, std::string* error) = 0;
    virtual bool PublishGaze(const GazeTargetFrame& frame, std::string* error) = 0;
    virtual void Close() = 0;
};

int RunRuntimeLoop(
    RuntimeBackend& backend,
    std::istream& input,
    std::ostream& output,
    std::ostream& diagnostics);

}  // namespace dds_bridge
}  // namespace visual_events
