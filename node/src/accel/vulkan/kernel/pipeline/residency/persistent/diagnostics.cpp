#include "../../../../kernel.hpp"

#include "../../../../../kernel/recurrence.hpp"

#include "../../../../map/dispatch.hpp"
#include "../../../../map/local.hpp"
#include "../../evidence.hpp"
#include "../../state.hpp"

#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace {

[[nodiscard]] bool
SameResidentRef(const rund::kernel::ResidentBufferRef &left,
                const rund::kernel::ResidentBufferRef &right) noexcept {
  return left.id == right.id && left.bytes == right.bytes &&
         left.offset_bytes == right.offset_bytes &&
         left.element_bytes == right.element_bytes &&
         left.stride_bytes == right.stride_bytes && left.count == right.count &&
         left.usage == right.usage;
}

[[nodiscard]] bool ValidHistoryOutput(const VulkanMapEncodeResources &map,
                                      const MapRecurrenceHistory &history,
                                      const std::uint64_t index) noexcept {
  const rund::kernel::ResidentBufferRef &ref = history.outputs[index];
  const std::shared_ptr<void> &owner = history.handles[index];
  const rund::kernel::ResidentBufferRef *const bound =
      map.bindings.resident_outputs.ref(index);
  const std::shared_ptr<void> *const bound_owner =
      map.bindings.resident_outputs.handle(index);
  if (bound == nullptr || bound_owner == nullptr || owner == nullptr ||
      *bound_owner != owner || !SameResidentRef(*bound, ref) || ref.id == 0u ||
      ref.bytes == 0u || ref.offset_bytes > ref.bytes || ref.count == 0u ||
      ref.element_bytes == 0u || ref.stride_bytes < ref.element_bytes ||
      ref.usage != rund::kernel::kResidentUsageWrite ||
      ref.count % map.iterations != 0u) {
    return false;
  }
  const std::uint64_t slice_count = ref.count / map.iterations;
  if (slice_count == 0u ||
      slice_count >
          std::numeric_limits<std::uint64_t>::max() / ref.stride_bytes ||
      history.pitch_bytes[index] != slice_count * ref.stride_bytes) {
    return false;
  }
  const std::uint64_t last = ref.count - 1u;
  if (last > std::numeric_limits<std::uint64_t>::max() / ref.stride_bytes ||
      ref.offset_bytes >
          std::numeric_limits<std::uint64_t>::max() - last * ref.stride_bytes ||
      ref.element_bytes > std::numeric_limits<std::uint64_t>::max() -
                              ref.offset_bytes - last * ref.stride_bytes ||
      ref.offset_bytes + last * ref.stride_bytes + ref.element_bytes >
          ref.bytes) {
    return false;
  }
  const VulkanResidentBufferResult &resident = map.resident.output(index);
  return resident.check.ok && resident.device_buffer != nullptr &&
         resident.handle == owner;
}

[[nodiscard]] bool
ValidVulkanHistory(const VulkanMapEncodeResources &map) noexcept {
  if (!map.history_recurrence || map.binding_owner == nullptr ||
      map.prepared == nullptr || map.prepared->plan.output_buffer_count == 0u ||
      map.iterations < 2u ||
      map.bindings.resident_outputs.count !=
          map.prepared->plan.output_buffer_count ||
      !map.bindings.resident_outputs.has_refs() ||
      !map.bindings.resident_outputs.has_handles() ||
      map.resident.bindings != &map.bindings ||
      map.resident.outputs.size() != map.prepared->plan.output_buffer_count) {
    return false;
  }
  const auto *const history =
      static_cast<const MapRecurrenceHistory *>(map.binding_owner.get());
  if (history == nullptr ||
      history->count != map.prepared->plan.output_buffer_count) {
    return false;
  }
  for (std::uint64_t index = 0u; index < map.prepared->plan.output_buffer_count;
       ++index) {
    if (!ValidHistoryOutput(map, *history, index)) {
      return false;
    }
  }
  return true;
}

} // namespace

bool InspectVulkanFusedDirectRecurrence(
    const std::shared_ptr<void> &prepared,
    VulkanFusedDirectRecurrenceDiagnostics &diagnostics) noexcept {
  diagnostics = {};
  const auto *const pipeline =
      static_cast<const VulkanPipeline *>(prepared.get());
  const auto *const map = pipeline == nullptr
                              ? nullptr
                              : static_cast<const VulkanMapEncodeResources *>(
                                    pipeline->recurrence.get());
  if (!ValidVulkanPipeline(pipeline) || map == nullptr ||
      map->adapter != pipeline->adapter || map->prepared == nullptr ||
      map->prepared->pipeline == nullptr ||
      (!map->history_recurrence && map->binding_owner != nullptr) ||
      map->controlled() || map->iterations < 2u || map->windows.size() != 1u ||
      map->windows.capacity() != 1u || map->descriptor_sets.count != 1u ||
      map->descriptor_sets.at(0u) == VK_NULL_HANDLE ||
      map->param.buffer.buffer == VK_NULL_HANDLE ||
      map->param.buffer.allocated_bytes == 0u ||
      pipeline->command.pool == VK_NULL_HANDLE ||
      pipeline->command.buffer == VK_NULL_HANDLE ||
      pipeline->command.fence == VK_NULL_HANDLE ||
      pipeline->dispatch_count != 1u || !pipeline->transducers.empty() ||
      (map->history_recurrence && !ValidVulkanHistory(*map))) {
    return false;
  }
  const std::uint64_t route_host_bytes = VulkanRecurrenceHostBytes(*pipeline);
  if (route_host_bytes == 0u) {
    return false;
  }
  diagnostics = VulkanFusedDirectRecurrenceDiagnostics{
      .iteration_count = map->iterations,
      .native_command_buffer_count = 1u,
      .native_dispatch_count = pipeline->dispatch_count,
      .descriptor_set_count = map->descriptor_sets.count,
      .device_buffer_bytes = map->param.buffer.allocated_bytes,
      .route_host_bytes = route_host_bytes,
      .push_constant_bytes = sizeof(VulkanMapDispatch),
      .command_buffer_allocation_bytes_observable = false,
      .retention = map->history_recurrence
                       ? VulkanFusedDirectRecurrenceRetention::History
                       : VulkanFusedDirectRecurrenceRetention::Terminal,
  };
  return true;
}

#endif

} // namespace rund::node::accel::detail
