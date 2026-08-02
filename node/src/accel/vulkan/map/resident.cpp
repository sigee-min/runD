#include "../../backend/number.hpp"
#include "../descriptor.hpp"
#include "local.hpp"

#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
bool VulkanMapResidentWindowSpan(
    const VulkanAdapter &adapter, const rund::kernel::ResidentBufferRef &ref,
    const rund::kernel::ComputeDispatchWindow &window, VkDeviceSize &offset,
    VkDeviceSize &range, const char *&reason) noexcept {
  if (ref.stride_bytes < ref.element_bytes) {
    reason = "compute_resident_stride_invalid";
    return false;
  }
  StorageRange planned{};
  if (!PlanStorage(adapter, ref, window.begin_sequence, window.tile_count,
                   planned)) {
    reason = "compute_resident_bytes_invalid";
    return false;
  }
  offset = planned.base;
  range = planned.bytes;
  reason = "ok";
  return true;
}

bool VulkanMapResidentOutputWindowSpan(
    const VulkanMapEncodeResources &map,
    const rund::kernel::ResidentBufferRef &ref,
    const rund::kernel::ComputeDispatchWindow &window, VkDeviceSize &offset,
    VkDeviceSize &range, const char *&reason) noexcept {
  if (!map.history_recurrence) {
    return map.adapter != nullptr &&
           VulkanMapResidentWindowSpan(*map.adapter, ref, window, offset,
                                       range, reason);
  }
  if (map.adapter == nullptr || map.iterations < 2u || ref.count == 0u ||
      ref.count % map.iterations != 0u) {
    reason = "compute_pipeline_recurrence_history_invalid";
    return false;
  }
  const std::uint64_t slice_count = ref.count / map.iterations;
  if (slice_count == 0u || window.tile_count == 0u ||
      window.begin_sequence > slice_count ||
      window.tile_count > slice_count - window.begin_sequence ||
      static_cast<std::uint64_t>(map.iterations - 1u) >
          std::numeric_limits<std::uint64_t>::max() / slice_count) {
    reason = "compute_pipeline_recurrence_history_invalid";
    return false;
  }
  const std::uint64_t previous_slices =
      static_cast<std::uint64_t>(map.iterations - 1u) * slice_count;
  if (window.tile_count >
      std::numeric_limits<std::uint64_t>::max() - previous_slices) {
    reason = "compute_pipeline_recurrence_history_invalid";
    return false;
  }
  const std::uint64_t span_count = previous_slices + window.tile_count;
  StorageRange planned{};
  if (!PlanStorage(*map.adapter, ref, window.begin_sequence, span_count,
                   planned)) {
    reason = "compute_resident_bytes_invalid";
    return false;
  }
  offset = planned.base;
  range = planned.bytes;
  reason = "ok";
  return true;
}

bool ValidateVulkanMapHistoryOutputs(const VulkanMapEncodeResources &map,
                                     const char *&reason) noexcept {
  reason = "ok";
  if (!map.history_recurrence) {
    return true;
  }
  if (map.adapter == nullptr || map.prepared == nullptr ||
      map.prepared->plan.output_buffer_count == 0u ||
      map.bindings.resident_outputs.count !=
          map.prepared->plan.output_buffer_count ||
      map.windows.empty()) {
    reason = "compute_pipeline_recurrence_history_invalid";
    return false;
  }
  for (const rund::kernel::ComputeDispatchWindow &window : map.windows) {
    for (std::uint64_t output = 0u;
         output < map.prepared->plan.output_buffer_count; ++output) {
      const rund::kernel::ResidentBufferRef *const ref =
          map.bindings.resident_outputs.ref(output);
      VkDeviceSize offset = 0u;
      VkDeviceSize range = 0u;
      if (ref == nullptr ||
          !VulkanMapResidentOutputWindowSpan(map, *ref, window, offset, range,
                                             reason)) {
        return false;
      }
    }
  }
  return true;
}
#endif

} // namespace rund::node::accel::detail
