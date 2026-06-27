#pragma once

#include "visual_events/dds_bridge/runtime_loop.hpp"
#include "visual_events/dds_bridge/runtime_options.hpp"

#include <memory>

namespace visual_events {
namespace dds_bridge {

void ConstructUnitreeChannelsOnce(const RuntimeOptions& options);
std::unique_ptr<RuntimeBackend> CreateUnitreeRuntimeBackend(const RuntimeOptions& options);

}  // namespace dds_bridge
}  // namespace visual_events
