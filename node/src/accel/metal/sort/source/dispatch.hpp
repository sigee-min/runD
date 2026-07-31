#pragma once

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] const char *MetalSortDispatchSource() {
  return R"MSL(
kernel void rund_compute_sort_dispatch(
    device uint* dispatch_args [[buffer(0)]],
    constant SortParams& params [[buffer(1)]],
    device const uint* logical_count [[buffer(2)]],
    device uint* status [[buffer(3)]]) {
  const SortRange range = rund_sort_range(logical_count, params);
  const ulong blocks =
      (range.logical + ulong(kSortBlockSize) - 1ul) / ulong(kSortBlockSize);
  dispatch_args[0] = range.invalid ? 0u : uint(blocks);
  dispatch_args[1] = 1u;
  dispatch_args[2] = 1u;
  status[0] = range.invalid ? 2u : 0u;
}
)MSL";
}

} // namespace
} // namespace rund::node::accel::detail
