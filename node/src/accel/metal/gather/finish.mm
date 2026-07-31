#include <accel/check.hpp>

#include "../../gather/status.hpp"
#include "local.hpp"

namespace rund::node::accel::detail {

bool ObserveMetalGatherFailure(const std::shared_ptr<void> &resources,
                               std::uint64_t &ordinal) noexcept {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  const auto *const gather =
      static_cast<const MetalGatherEncodeResources *>(resources.get());
  const auto *const status = gather == nullptr
                                 ? nullptr
                                 : static_cast<const std::uint32_t *>(
                                       MetalBufferContents(gather->status));
  if (status == nullptr || gather->status.bytes < 2u * sizeof(*status) ||
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

rund::AccelCheck FinishMetalGather(MetalAdapter &adapter,
                                   const std::shared_ptr<void> &resources) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  auto *const gather =
      static_cast<MetalGatherEncodeResources *>(resources.get());
  if (gather == nullptr || gather->adapter != &adapter) {
    SetMetalLastError(adapter, "compute_gather_invalid");
    return rund::AccelCheck{false, "compute_gather_invalid"};
  }
  const auto *const status = static_cast<const rund::kernel::u32 *>(
      MetalBufferContents(gather->status));
  if (status == nullptr) {
    SetMetalLastError(adapter, "accel_metal_buffer_unavailable");
    return rund::AccelCheck{false, "accel_metal_buffer_unavailable"};
  }
  const rund::AccelCheck check = GatherStatus(status[0]);
  if (!check.ok) {
    SetMetalLastError(adapter, check.reason);
    return check;
  }
  RecordMetalDispatches(adapter, gather->plan.pass_count);
  SetMetalLastError(adapter, "ok");
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
