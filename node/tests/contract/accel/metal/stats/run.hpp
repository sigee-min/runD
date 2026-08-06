#pragma once

#include <accel/device.hpp>
#include <kernel/program/compute/plan.hpp>

#include "bindings.hpp"

#include "../../runtime/local/windows.hpp"
#include "src/accel/metal/stats.hpp"

namespace node_accel_contract {

[[nodiscard]] bool
MetalRepeatedStagedRunsReportWarmRuntimeStats(const rund::AccelDevice &pick) {
  if (!pick.check.ok || !pick.caps.ok || !pick.backend) {
    return true;
  }

  metal_stats::Work work{};
  const metal_stats::NodeComputeOp op = metal_stats::BuildOp(pick.caps.api);
  const rund::kernel::ComputePlan plan =
      rund::kernel::PlanCompute(metal_stats::Phase(), op.map, pick.caps,
                                rund::kernel::ComputeLimit{
                                    .staging_bytes = pick.caps.staging_bytes,
                                    .max_window_tiles = 2u,
                                });
  if (!plan.ok || plan.dispatch_window_tiles == 0u) {
    return false;
  }

  auto input_buffers = metal_stats::InputBuffers(work);
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir, plan.api);
  const std::vector<rund::kernel::ComputeDispatchWindow> windows =
      runtime_case::DispatchWindows(plan);
  if (windows.empty()) {
    return false;
  }

  rund::node::accel::detail::ResetMetalRuntimeStats(pick);
  const rund::kernel::BindingSet first =
      metal_stats::BindingsFor(op, pick, work, input_buffers, work.first);
  if (!pick.backend.execute(pick.backend.context, plan, artifact,
                            windows.data(), windows.size(), first)) {
    return false;
  }
  const rund::kernel::BindingSet second =
      metal_stats::BindingsFor(op, pick, work, input_buffers, work.second);
  if (!pick.backend.execute(pick.backend.context, plan, artifact,
                            windows.data(), windows.size(), second)) {
    return false;
  }

  const rund::node::accel::detail::MetalRuntimeStats stats =
      rund::node::accel::detail::ReadMetalRuntimeStats(pick);
  const std::uint64_t expected_host_to_device_bytes =
      (plan.param_bytes + plan.input_bytes_per_tile * plan.tile_count) * 2u;
  return work.first == work.expected && work.second == work.expected &&
         stats.dispatch_count != 0u && stats.pipeline_compile_count != 0u &&
         stats.pipeline_cache_hit_count >= 1u &&
         stats.buffer_reuse_hit_count != 0u &&
         stats.host_to_device_bytes == expected_host_to_device_bytes &&
         stats.device_to_host_bytes >=
             plan.output_bytes_per_tile * plan.tile_count * 2u;
}

} // namespace node_accel_contract
