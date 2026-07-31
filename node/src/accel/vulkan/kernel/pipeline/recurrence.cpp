#include "recurrence.hpp"

#include "../../buffer/create/telemetry.hpp"
#include "../../map/api.hpp"
#include "../../map/local.hpp"

#include <limits>
#include <new>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] rund::AccelCheck PrepareVulkanRecurrence(
    const std::span<const BackendBatchEntry> entries,
    const MapRecurrence &recurrence, PreparedPipelineStatusLayout &status,
    VulkanPipeline &pipeline, PreparedMemory &staging_memory) {
  if (!recurrence.ready() || recurrence.first == nullptr ||
      recurrence.windows == nullptr || recurrence.window_count == 0u ||
      recurrence.iterations < 2u ||
      recurrence.iterations > std::numeric_limits<std::uint32_t>::max()) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  for (std::size_t index = 0u; index < entries.size(); ++index) {
    const BackendBatchEntry &entry = entries[index];
    VulkanKernelContext context{};
    const rund::AccelCheck valid =
        entry.run == nullptr || entry.run->pick == nullptr
            ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
            : ValidateVulkanKernelContext(*entry.run->pick, context);
    if (!valid.ok || context.adapter == nullptr ||
        (pipeline.adapter != nullptr && pipeline.adapter != context.adapter) ||
        !SetPreparedProgramStatusSlice(status,
                                       static_cast<std::uint32_t>(index), 0u)) {
      return valid.ok ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
                      : valid;
    }
    pipeline.adapter = context.adapter;
  }
  if (pipeline.adapter == nullptr || recurrence.first->control.active()) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  std::scoped_lock lock{pipeline.adapter->mutex};
  const VulkanMemoryStats before = pipeline.adapter->staging_memory;
  std::shared_ptr<void> resource{};
  rund::AccelCheck prepared{};
  try {
    prepared = PrepareVulkanMap(*entries.front().run->pick, recurrence.plan,
                                recurrence.artifact, recurrence.windows,
                                recurrence.window_count, recurrence.bindings,
                                recurrence.first->control, resource);
  } catch (const std::bad_alloc &) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  auto *const map = static_cast<VulkanMapEncodeResources *>(resource.get());
  if (!prepared.ok || map == nullptr || map->control.has_count() ||
      map->control.has_predicate() || map->windows.empty() ||
      map->plan.dispatch_count != map->windows.size()) {
    return prepared.ok ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
                       : prepared;
  }
  map->iterations = static_cast<std::uint32_t>(recurrence.iterations);
  pipeline.dispatch_count = map->windows.size();
  pipeline.recurrence = std::move(resource);
  staging_memory =
      VulkanPreparedMemory(before, pipeline.adapter->staging_memory,
                           pipeline.adapter->caps.staging_bytes);
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
