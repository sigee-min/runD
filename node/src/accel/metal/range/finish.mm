#include <accel/check.hpp>

#include "local.hpp"

#include <string_view>

namespace rund::node::accel::detail {

rund::AccelCheck FinishMetalRange(MetalAdapter &adapter,
                                  const std::shared_ptr<void> &resources) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  auto *const range = static_cast<MetalRangeResources *>(resources.get());
  if (range == nullptr || range->adapter != &adapter) {
    SetMetalLastError(adapter, "compute_range_aggregate_invalid");
    return rund::AccelCheck{false, "compute_range_aggregate_invalid"};
  }
  std::uint64_t dispatches = range->range.stage_count();
  if (range->controlled) {
    const auto *const status = static_cast<const std::uint32_t *>(
        MetalBufferContents(range->control_status));
    const auto *const indirect = static_cast<const RangeIndirect *>(
        MetalBufferContents(range->control_indirect));
    if (status == nullptr || indirect == nullptr) {
      return rund::AccelCheck{false, "accel_metal_buffer_unavailable"};
    }
    if (status[0] != 0u) {
      SetMetalLastError(adapter, "compute_workset_overflow");
      return rund::AccelCheck{false, "compute_workset_overflow"};
    }
    dispatches = 1u;
    for (std::size_t index = 0u; index < range->stage_count; ++index) {
      dispatches += static_cast<std::uint64_t>(indirect[index].groups_x != 0u);
    }
  }
  RecordMetalDispatches(adapter, dispatches);
  SetMetalLastError(adapter, "ok");
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
