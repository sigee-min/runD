#pragma once

#include <accel/check.hpp>

#include "../local.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] bool
ObserveMetalMapFailure(const std::shared_ptr<void> &resources,
                       std::uint64_t &ordinal) noexcept {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  const auto *const map =
      static_cast<const MetalMapEncodeResources *>(resources.get());
  const auto *const status =
      map == nullptr || map->checks.empty()
          ? nullptr
          : static_cast<const std::uint32_t *>(
                MetalBufferContents(map->control_status));
  if (status == nullptr || map->control_status.bytes < 2u * sizeof(*status) ||
      status[0] != 2u) {
    return false;
  }
  ordinal = status[1];
  return true;
#else
  (void)resources;
  (void)ordinal;
  return false;
#endif
}

rund::AccelCheck FinishMetalMap(MetalAdapter &adapter,
                                const std::shared_ptr<void> &resources) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  auto *const map = static_cast<MetalMapEncodeResources *>(resources.get());
  if (map == nullptr || map->adapter != &adapter) {
    SetMetalLastError(adapter, "compute_plan_invalid");
    return rund::AccelCheck{false, "compute_plan_invalid"};
  }
  if (map->controlled()) {
    const auto *const status = static_cast<const std::uint32_t *>(
        MetalBufferContents(map->control_status));
    if (status == nullptr) {
      SetMetalLastError(adapter, "accel_metal_buffer_unavailable");
      return rund::AccelCheck{false, "accel_metal_buffer_unavailable"};
    }
    if (status[0] != 0u) {
      const char *const reason = status[0] == 2u
                                     ? "compute_gather_index_out_of_range"
                                     : "compute_workset_overflow";
      SetMetalLastError(adapter, reason);
      return rund::AccelCheck{false, reason};
    }
  }
  RecordMetalDispatches(adapter, map->windows.size());
  SetMetalLastError(adapter, "ok");
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
