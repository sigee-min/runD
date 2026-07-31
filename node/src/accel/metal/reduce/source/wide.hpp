#pragma once

#include "op.hpp"

#include <string>

namespace rund::node::accel::detail {

inline void AppendMetalReduceU64(std::string &source,
                                 const rund::kernel::ReduceOp op,
                                 const bool signed_domain,
                                 const char *const op_name) {
  source += "rund_compute_reduce_";
  source += op_name;
  source += R"MSL(_u64(
    device const ulong* input [[buffer(0)]],
    device ulong* partial [[buffer(1)]],
    device ulong* output [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]],
    constant ReduceParams& params [[buffer(4)]],
    device const uint* logical_count [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  threadgroup ulong sums[RUND_REDUCE_BLOCK_SIZE];
  threadgroup uint overflows[RUND_REDUCE_BLOCK_SIZE];
  const ulong local = ulong(tid);
  const ulong index = ulong(group) * ulong(RUND_REDUCE_BLOCK_SIZE) + local;
  const ulong resident_count = params.count_words == 2u
      ? (ulong(logical_count[1]) << 32u) | ulong(logical_count[0])
      : (params.count_words == 1u ? ulong(logical_count[0]) : params.input_count);
  const ulong active_count = params.initial_pass != 0u
      ? min(resident_count, params.input_count) : params.input_count;
  if (params.initial_pass != 0u && resident_count > params.input_count && tid == 0u && group == 0u) {
    atomic_fetch_or_explicit(&status[0], 2u, memory_order_relaxed);
  }
)MSL";
  if (op == rund::kernel::ReduceOp::Min || op == rund::kernel::ReduceOp::Max) {
    source += "  if (params.initial_pass != 0u && active_count == 0ul && "
              "tid == 0u && group == 0u) { atomic_fetch_or_explicit("
              "&status[0], 4u, memory_order_relaxed); }\n";
  }
  source += MetalReduceInitU64(op, signed_domain);
  source += R"MSL(
  overflows[tid] = 0u;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint width = RUND_REDUCE_BLOCK_SIZE;
  while (width > 1u) {
    const uint next = (width + 1u) >> 1u;
    if (tid < width - next) {
      const ulong rhs = sums[tid + next];
)MSL";
  source += MetalReduceCombine(op, signed_domain, true);
  source += MetalReduceOverflowU64(op, signed_domain);
  source += R"MSL(
      overflows[tid] |= overflows[tid + next];
      sums[tid] = combined;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    width = next;
  }
  if (tid == 0u) {
    if (overflows[0] != 0u) { atomic_fetch_or_explicit(&status[0], 1u, memory_order_relaxed); }
    if (params.final_pass != 0u) { output[0] = sums[0]; }
    else { partial[params.output_offset + ulong(group)] = sums[0]; }
  }
}
)MSL";
}

} // namespace rund::node::accel::detail
