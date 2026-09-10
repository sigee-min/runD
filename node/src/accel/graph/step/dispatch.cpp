#include "dispatch.hpp"

#include <kernel/program/compute/plan.hpp>

#include <limits>

namespace rund::node::accel::detail::step {

FrozenDispatchCount
CountMapDispatch(const rund::kernel::ExecutionMetadata &metadata,
                 const std::uint64_t element_count,
                 const rund::kernel::ComputeCaps &caps,
                 const std::uint64_t phase_id) {
  if (!metadata.ok || element_count == 0u || phase_id == 0u) {
    return FrozenDispatchCount{.reason = "accel_kernel_graph_invalid"};
  }
  const rund::kernel::ComputeLimit limit{
      .staging_bytes = caps.staging_bytes,
      .max_window_tiles = caps.max_window_tiles,
  };
  rund::kernel::ComputeMap map = metadata.map;
  map.api = caps.api;
  const rund::kernel::TilePhaseDescription phase{
      .phase_id = phase_id,
      .tile_count = element_count,
  };
  const rund::kernel::ComputeDispatchPlan dispatch =
      rund::kernel::PlanComputeDispatch(phase, map, caps, limit);
  if (!dispatch.ok || dispatch.dispatch_count == 0u) {
    return FrozenDispatchCount{.reason = dispatch.reason};
  }
  return FrozenDispatchCount{
      .count = dispatch.dispatch_count,
      .ok = true,
      .reason = "ok",
  };
}

FrozenDispatchCount
CountOriginalDispatch(const std::span<const GraphCompileNode> nodes,
                      const rund::kernel::ComputeCaps &caps,
                      const std::uint64_t phase_offset) {
  FrozenDispatchCount result{};
  for (std::size_t index = 0u; index < nodes.size(); ++index) {
    const GraphCompileNode &node = nodes[index];
    if (node.kind() != rund::kernel::NodeKind::Map) {
      result.reason = "accel_kernel_graph_invalid";
      return result;
    }
    const FrozenDispatchCount dispatch =
        CountMapDispatch(node.map_metadata, node.element_count, caps,
                         phase_offset + static_cast<std::uint64_t>(index) + 1u);
    if (!dispatch.ok) {
      result.reason = dispatch.reason;
      return result;
    }
    if (result.count >
        std::numeric_limits<std::uint64_t>::max() - dispatch.count) {
      result.reason = "compute_dispatch_overflow";
      return result;
    }
    result.count += dispatch.count;
  }
  result.ok = result.count != 0u;
  result.reason = result.ok ? "ok" : "compute_plan_invalid";
  return result;
}

} // namespace rund::node::accel::detail::step
