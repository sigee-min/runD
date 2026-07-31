#include "recurrence.hpp"

#include "../../buffer/create/telemetry.hpp"
#include "../../map/api.hpp"
#include "../../map/local.hpp"

#include <limits>
#include <new>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] bool ValidRecurrence(const MapRecurrence &recurrence) noexcept {
  return recurrence.ready() && recurrence.first != nullptr &&
         recurrence.windows != nullptr && recurrence.window_count != 0u &&
         recurrence.iterations >= 2u &&
         recurrence.iterations <= std::numeric_limits<std::uint32_t>::max();
}

[[nodiscard]] rund::AccelCheck PrepareRecurrenceMap(
    const BackendRun &owner, const MapRecurrence &recurrence,
    VulkanAdapter &adapter, std::shared_ptr<void> &resource) {
  if (!ValidRecurrence(recurrence) || owner.pick == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  std::shared_ptr<void> prepared;
  const rund::AccelCheck ready = PrepareVulkanMap(
      *owner.pick, recurrence.plan, recurrence.artifact, recurrence.windows,
      recurrence.window_count, recurrence.bindings, recurrence.first->control,
      prepared);
  auto *const map = static_cast<VulkanMapEncodeResources *>(prepared.get());
  if (!ready.ok || map == nullptr || map->adapter != &adapter ||
      map->controlled() || map->windows.empty() ||
      map->plan.dispatch_count != map->windows.size()) {
    return ready.ok ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
                    : ready;
  }
  map->iterations = static_cast<std::uint32_t>(recurrence.iterations);
  resource = std::move(prepared);
  return rund::AccelCheck{true, "ok"};
}

} // namespace

[[nodiscard]] rund::AccelCheck PrepareVulkanRecurrence(
    const std::span<const BackendBatchEntry> entries,
    const MapRecurrence &recurrence, PreparedPipelineStatusLayout &status,
    VulkanPipeline &pipeline, PreparedMemory &staging_memory) {
  if (!ValidRecurrence(recurrence)) {
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
  rund::AccelCheck ready{};
  try {
    ready = PrepareRecurrenceMap(*entries.front().run, recurrence,
                                 *pipeline.adapter, pipeline.recurrence);
  } catch (const std::bad_alloc &) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  if (!ready.ok) {
    return ready;
  }
  pipeline.dispatch_count = recurrence.window_count;
  staging_memory =
      VulkanPreparedMemory(before, pipeline.adapter->staging_memory,
                           pipeline.adapter->caps.staging_bytes);
  return rund::AccelCheck{true, "ok"};
}

rund::AccelCheck PrepareVulkanTransducers(
    const std::span<const BackendBatchEntry> templates,
    const std::span<const TileTransducer> transducers, VulkanPipeline &pipeline,
    PreparedMemory &staging_memory) {
  staging_memory = {};
  pipeline.transducers.clear();
  if (transducers.empty()) {
    return rund::AccelCheck{true, "ok"};
  }
  if (pipeline.adapter == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  try {
    pipeline.transducers.resize(transducers.size());
  } catch (const std::bad_alloc &) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }

  for (const TileTransducer &transducer : transducers) {
    const MapRecurrence &recurrence = transducer.recurrence;
    if (!ValidRecurrence(recurrence) ||
        recurrence.iterations != transducer.template_count ||
        transducer.template_first >= templates.size() ||
        transducer.template_count >
            templates.size() - transducer.template_first) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    const BackendBatchEntry &owner = templates[transducer.template_first];
    VulkanKernelContext context{};
    const rund::AccelCheck valid =
        owner.run == nullptr || owner.run->pick == nullptr
            ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
            : ValidateVulkanKernelContext(*owner.run->pick, context);
    if (!valid.ok || context.adapter != pipeline.adapter) {
      return valid.ok ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
                      : valid;
    }
  }

  std::scoped_lock lock{pipeline.adapter->mutex};
  const VulkanMemoryStats before = pipeline.adapter->staging_memory;
  try {
    for (std::size_t index = 0u; index < transducers.size(); ++index) {
      const TileTransducer &transducer = transducers[index];
      const BackendBatchEntry &owner =
          templates[transducer.template_first];
      const rund::AccelCheck ready =
          PrepareRecurrenceMap(*owner.run, transducer.recurrence,
                               *pipeline.adapter, pipeline.transducers[index]);
      if (!ready.ok) {
        pipeline.transducers.clear();
        return ready;
      }
    }
  } catch (const std::bad_alloc &) {
    pipeline.transducers.clear();
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  staging_memory =
      VulkanPreparedMemory(before, pipeline.adapter->staging_memory,
                           pipeline.adapter->caps.staging_bytes);
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
