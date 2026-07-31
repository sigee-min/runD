#include "local.hpp"

#include <string>

namespace rund::node::accel::detail {

std::string MetalScatterSource() {
  return R"MSL(
#include <metal_stdlib>
using namespace metal;

struct ScatterParams {
  ulong element_count;
  ulong output_count;
};

inline void rund_scatter_record_failure(device atomic_uint* status,
                                        uint gid,
                                        uint reason) {
  atomic_fetch_min_explicit(&status[0], (gid << 1u) | reason,
                            memory_order_relaxed);
}

inline bool rund_scatter_claim_target(device atomic_uint* status,
                                      uint gid,
                                      uint target) {
  const uint prior =
      atomic_fetch_min_explicit(&status[ulong(target) + 1ul], gid,
                                memory_order_relaxed);
  if (prior == 0xffffffffu) { return true; }
  const uint duplicate = prior < gid ? gid : prior;
  rund_scatter_record_failure(status, duplicate, 1u);
  return false;
}

kernel void rund_compute_scatter_u32(
    device const uint* values [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device uint* output [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]],
    constant ScatterParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  if (ulong(gid) >= params.element_count) { return; }
  const uint target = indices[gid];
  if (ulong(target) >= params.output_count) {
    rund_scatter_record_failure(status, gid, 0u);
    return;
  }
  if (rund_scatter_claim_target(status, gid, target)) {
    output[target] = values[gid];
  }
}

kernel void rund_compute_scatter_u64(
    device const ulong* values [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device ulong* output [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]],
    constant ScatterParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  if (ulong(gid) >= params.element_count) { return; }
  const uint target = indices[gid];
  if (ulong(target) >= params.output_count) {
    rund_scatter_record_failure(status, gid, 0u);
    return;
  }
  if (rund_scatter_claim_target(status, gid, target)) {
    output[target] = values[gid];
  }
}
)MSL";
}

} // namespace rund::node::accel::detail
