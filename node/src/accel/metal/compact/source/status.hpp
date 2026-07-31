#pragma once

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] const char* MetalCompactStatusSource() {
  return R"MSL(
kernel void rund_compute_compact_status(
    device const uint* flags [[buffer(0)]],
    device const uint* offsets [[buffer(1)]],
    device const uint* scan_totals [[buffer(2)]],
    device uint* status [[buffer(3)]],
    constant CompactParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid != 0u) {
    return;
  }
  const ulong last = params.element_count - 1ul;
  uint selected = offsets[last] + (flags[last] == 0u ? 0u : 1u);
  if (params.scan_block_size != 0u) {
    selected += scan_totals[last / ulong(params.scan_block_size)];
  }
  status[0] = selected;
}
)MSL";
}

}  // namespace
}  // namespace rund::node::accel::detail
