#pragma once
#include "../local.hpp"
namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline const char* ReduceFunctionName(
    const rund::kernel::ReduceOp op,
    const rund::kernel::ReduceElement element) noexcept {
  if (element == rund::kernel::ReduceElement::U64) {
    return op == rund::kernel::ReduceOp::CountNonzero
               ? "rund_compute_reduce_count_nonzero_u64"
           : op == rund::kernel::ReduceOp::Min ? "rund_compute_reduce_min_u64"
           : op == rund::kernel::ReduceOp::Max ? "rund_compute_reduce_max_u64"
                                               : "rund_compute_reduce_sum_u64";
  }
  return op == rund::kernel::ReduceOp::CountNonzero
             ? "rund_compute_reduce_count_nonzero_u32"
         : op == rund::kernel::ReduceOp::Min ? "rund_compute_reduce_min_u32"
         : op == rund::kernel::ReduceOp::Max ? "rund_compute_reduce_max_u32"
                                             : "rund_compute_reduce_sum_u32";
}
#endif
}  // namespace rund::node::accel::detail
