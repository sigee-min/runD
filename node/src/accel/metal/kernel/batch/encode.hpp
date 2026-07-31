#pragma once

#include "model.hpp"

#include "../../command/run.hpp"

#include <accel/check.hpp>

#include <span>

namespace rund::node::accel::detail {
struct BackendBatchEntry;
} // namespace rund::node::accel::detail

namespace rund::node::accel::detail::metalbatch {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] rund::AccelCheck
encode(MetalAdapter &adapter, std::span<const BackendBatchEntry> entries,
       std::span<const BatchMapView> views, const BatchMapPlan &plan,
       const Maps &maps, const Workspace *workspace, CommandRun &command);
#endif

} // namespace rund::node::accel::detail::metalbatch
