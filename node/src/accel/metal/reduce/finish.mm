#include <accel/check.hpp>

#include "../../reduce/metal.hpp"
#include "local.hpp"

namespace rund::node::accel::detail {

rund::AccelCheck FinishMetalReduce(MetalAdapter &adapter,
                                   const std::shared_ptr<void> &resources) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  auto *const reduce =
      static_cast<MetalReduceEncodeResources *>(resources.get());
  if (reduce == nullptr || reduce->adapter != &adapter) {
    SetMetalLastError(adapter, "compute_reduce_invalid");
    return rund::AccelCheck{false, "compute_reduce_invalid"};
  }
  const auto *const status = static_cast<const rund::kernel::u32 *>(
      MetalBufferContents(reduce->status));
  if (status == nullptr) {
    SetMetalLastError(adapter, "accel_metal_buffer_unavailable");
    return rund::AccelCheck{false, "accel_metal_buffer_unavailable"};
  }
  if (*status != 0u) {
    const char *const reason =
        (*status & 2u) != 0u
            ? "compute_bounded_count_invalid"
            : ((*status & 4u) != 0u
                   ? "compute_reduce_count_zero"
                   : (reduce->plan.op == rund::kernel::ReduceOp::CountNonzero
                          ? "compute_reduce_count_overflow"
                          : "compute_reduce_sum_overflow"));
    SetMetalLastError(adapter, reason);
    return rund::AccelCheck{false, reason};
  }
  RecordMetalDispatches(adapter, reduce->plan.pass_count);
  SetMetalLastError(adapter, "ok");
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
