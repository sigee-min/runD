#include <accel/check.hpp>

#include "local.hpp"

namespace rund::node::accel::detail {

rund::AccelCheck FinishMetalSort(MetalAdapter &adapter,
                                 const std::shared_ptr<void> &resources) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  auto *const sort = static_cast<MetalSortEncodeResources *>(resources.get());
  if (sort == nullptr || sort->adapter != &adapter) {
    SetMetalLastError(adapter, "compute_sort_invalid");
    return rund::AccelCheck{false, "compute_sort_invalid"};
  }
  const bool bounded = sort->plan.count_source !=
                       rund::kernel::ComputeCountSource::Descriptor;
  RecordMetalDispatches(adapter, (bounded ? 1u : 0u) +
                                     4u * sort->plan.radix_pass_count);
  if (bounded) {
    const auto *const status = static_cast<const rund::kernel::u32 *>(
        MetalBufferContents(sort->status));
    if (status == nullptr) {
      SetMetalLastError(adapter, "accel_metal_buffer_unavailable");
      return rund::AccelCheck{false, "accel_metal_buffer_unavailable"};
    }
    if (*status != 0u) {
      const char *const reason = "compute_bounded_count_invalid";
      SetMetalLastError(adapter, reason);
      return rund::AccelCheck{false, reason};
    }
  }
  SetMetalLastError(adapter, "ok");
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
