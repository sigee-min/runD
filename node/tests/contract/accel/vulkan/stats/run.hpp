#pragma once

#include <accel/device.hpp>
#include <kernel/program/compute/plan.hpp>
#include <accel/runtime.hpp>

#include <node/accel/pick.hpp>

#include <node/accel/buffer.hpp>

#include "../../runtime/local/windows.hpp"
#include "bindings.hpp"

#include <cstdio>

namespace node_accel_contract {

[[nodiscard]] bool VulkanRepeatedStagedRunsReportWarmRuntimeStats() {
  const rund::AccelDevice pick =
      rund::node::accel::PickAccel(vulkan_stats::Policy());
  if (!pick.check.ok) {
    return vulkan::FailureReasonIsPrecise(pick);
  }

  vulkan_stats::Work work{};
  const rund::compute_dsl::ComputeOp op = vulkan_stats::BuildOp(work);
  rund::kernel::ComputeMap map = op.map();
  map.api = rund::kernel::ComputeApi::Vulkan;
  const rund::kernel::ComputePlan plan =
      rund::kernel::PlanCompute(vulkan_stats::Phase(), map, pick.caps,
                                rund::kernel::ComputeLimit{
                                    .staging_bytes = pick.caps.staging_bytes,
                                    .max_window_tiles = 2u,
                                });
  if (!plan.ok || plan.dispatch_window_tiles == 0u) {
    std::fprintf(stderr,
                 "vulkan stats plan failed ok=%d reason=%s windows=%llu\n",
                 plan.ok ? 1 : 0, plan.reason,
                 static_cast<unsigned long long>(plan.dispatch_window_tiles));
    return false;
  }

  auto input_buffers = vulkan_stats::InputBuffers(work);
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir(), plan.api);
  const std::vector<rund::kernel::ComputeDispatchWindow> windows =
      runtime_case::DispatchWindows(plan);
  if (windows.empty()) {
    return false;
  }

  rund::node::accel::ResetRuntimeStats(pick);
  const rund::kernel::BindingSet first =
      vulkan_stats::BindingsFor(map, work, input_buffers, work.first);
  if (!pick.backend.execute(pick.backend.context, plan, artifact,
                            windows.data(), windows.size(), first)) {
    std::fprintf(stderr, "vulkan stats first execute failed\n");
    return false;
  }
  std::array<rund::kernel::u32, 4u> first_values{};
  for (std::size_t index = 0u; index < first_values.size(); ++index) {
    first_values[index] = work.first[index].value;
  }
  const rund::RuntimeStats first_stats =
      rund::node::accel::ReadRuntimeStats(pick);
  if (first_values != work.expected || !first_stats.ok ||
      first_stats.pipeline_compile_count == 0u ||
      first_stats.descriptor_pool_create_count == 0u ||
      first_stats.descriptor_set_allocate_count == 0u) {
    std::fprintf(
        stderr,
        "vulkan stats first mismatch values=%d stats_ok=%d compile=%llu "
        "pool=%llu set=%llu actual0=%u expected0=%u\n",
        first_values == work.expected ? 1 : 0, first_stats.ok ? 1 : 0,
        static_cast<unsigned long long>(first_stats.pipeline_compile_count),
        static_cast<unsigned long long>(
            first_stats.descriptor_pool_create_count),
        static_cast<unsigned long long>(
            first_stats.descriptor_set_allocate_count),
        first_values[0], work.expected[0]);
    return false;
  }

  rund::node::accel::ResetRuntimeStats(pick);
  const rund::kernel::BindingSet second =
      vulkan_stats::BindingsFor(map, work, input_buffers, work.second);
  if (!pick.backend.execute(pick.backend.context, plan, artifact,
                            windows.data(), windows.size(), second)) {
    std::fprintf(stderr, "vulkan stats second execute failed\n");
    return false;
  }

  std::array<rund::kernel::u32, 4u> second_values{};
  for (std::size_t index = 0u; index < second_values.size(); ++index) {
    second_values[index] = work.second[index].value;
  }
  const rund::RuntimeStats stats = rund::node::accel::ReadRuntimeStats(pick);
  const bool ok =
      second_values == work.expected && stats.ok &&
      stats.dispatch_count != 0u && stats.pipeline_compile_count == 0u &&
      stats.pipeline_cache_hit_count >= 1u &&
      stats.descriptor_pool_create_count == 0u &&
      stats.descriptor_set_allocate_count == 0u &&
      stats.descriptor_reuse_hit_count >= 1u &&
      stats.host_to_device_bytes ==
          plan.param_bytes + plan.input_bytes_per_tile * plan.tile_count &&
      stats.device_to_host_bytes ==
          plan.output_bytes_per_tile * plan.tile_count &&
      stats.buffer_reuse_hit_count != 0u;
  if (!ok) {
    std::fprintf(
        stderr,
        "vulkan stats second mismatch values=%d stats=%d dispatch=%llu "
        "compile=%llu cache=%llu pool=%llu alloc=%llu reuse=%llu h2d=%llu/%llu "
        "d2h=%llu/%llu buffer=%llu\n",
        second_values == work.expected ? 1 : 0, stats.ok ? 1 : 0,
        static_cast<unsigned long long>(stats.dispatch_count),
        static_cast<unsigned long long>(stats.pipeline_compile_count),
        static_cast<unsigned long long>(stats.pipeline_cache_hit_count),
        static_cast<unsigned long long>(stats.descriptor_pool_create_count),
        static_cast<unsigned long long>(stats.descriptor_set_allocate_count),
        static_cast<unsigned long long>(stats.descriptor_reuse_hit_count),
        static_cast<unsigned long long>(stats.host_to_device_bytes),
        static_cast<unsigned long long>(
            plan.param_bytes + plan.input_bytes_per_tile * plan.tile_count),
        static_cast<unsigned long long>(stats.device_to_host_bytes),
        static_cast<unsigned long long>(plan.output_bytes_per_tile *
                                        plan.tile_count),
        static_cast<unsigned long long>(stats.buffer_reuse_hit_count));
  }
  return ok;
}

} // namespace node_accel_contract
