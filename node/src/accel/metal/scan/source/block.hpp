#pragma once

#include "../../../kernel/backend/source/sink.hpp"

namespace rund::node::accel::detail {

template <bool Wide, typename Sink>
[[nodiscard]] bool AppendMetalScanBlockSource(Sink &sink) noexcept(
    noexcept(sink.append(std::string_view{}))) {
  backend_source_recipe::SourceBuilder<Sink> source{sink};
  constexpr std::string_view type = Wide ? "ulong" : "uint";
  constexpr std::string_view zero = Wide ? "0ul" : "0u";
  constexpr std::string_view suffix = Wide ? "u64" : "u32";
  source += R"MSL(
kernel void rund_compute_scan_block_)MSL";
  source += suffix;
  source += R"MSL((
    device const )MSL";
  source += type;
  source += R"MSL(* input [[buffer(0)]],
    device )MSL";
  source += type;
  source += R"MSL(* output [[buffer(1)]],
    device )MSL";
  source += type;
  source += R"MSL(* totals [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]],
    constant ulong& element_count [[buffer(4)]],
    constant ulong& block_size [[buffer(5)]],
    constant uint& inclusive [[buffer(6)]],
    device const uint* logical_count [[buffer(7)]],
    constant uint& count_words [[buffer(8)]],
    constant uint& signed_domain [[buffer(9)]],
    constant uint& canonical_here [[buffer(10)]],
    uint physical_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]],
    uint width [[threads_per_threadgroup]],
    uint block [[threadgroup_position_in_grid]]) {
  const uint tid = simd_group * simd_width +
      simd_prefix_exclusive_sum(1u);
)MSL";
  if constexpr (Wide) {
    source += R"MSL(  threadgroup ulong values[kScanWidth + 1u];
)MSL";
  } else {
    source += R"MSL(  threadgroup uint values[2][kScanWidth];
)MSL";
  }
  source += R"MSL(  const ulong begin = ulong(block) * block_size;
  const ulong resident_count = count_words == 2u
      ? (ulong(logical_count[1]) << 32u) | ulong(logical_count[0])
      : (count_words == 1u ? ulong(logical_count[0]) : element_count);
  const ulong active_count = min(resident_count, element_count);
  const ulong end = min(begin + block_size, active_count);
  const ulong lane_size =
      (block_size + ulong(width) - 1ul) / ulong(width);
  const ulong lane_begin = min(begin + ulong(tid) * lane_size, end);
  const ulong lane_end = min(lane_begin + lane_size, end);
  )MSL";
  source += type;
  source += R"MSL( running = )MSL";
  source += zero;
  source += R"MSL(;
  for (ulong index = lane_begin; index < lane_end; ++index) {
    running += input[index];
  }
)MSL";
  if constexpr (!Wide) {
    source += R"MSL(  values[0][tid] = running;
)MSL";
  }
  source += R"MSL(  )MSL";
  source += type;
  source += R"MSL( offset = )MSL";
  source += zero;
  source += R"MSL(;
)MSL";
  if constexpr (Wide) {
    source += R"MSL(  const )MSL";
    source += type;
    source += R"MSL( total = rund_scan_exclusive_ulong(
      values, running, tid, width, simd_width, physical_lane, offset);
)MSL";
  } else {
    source += R"MSL(  )MSL";
    source += type;
    source += R"MSL( total = )MSL";
    source += zero;
    source += R"MSL(;
  rund_scan_exclusive_uint_pair(values[0], values[1], tid, width, simd_width, offset,
                                total);
)MSL";
  }
  source += R"MSL(  running = offset;
  uint bad = 0u;
  for (ulong index = lane_begin; index < lane_end; ++index) {
    const )MSL";
  source += type;
  source += R"MSL( value = input[index];
    const )MSL";
  source += type;
  source += R"MSL( next = running + value;
    output[index] = canonical_here != 0u && inclusive == 0u ? running : next;
    if ((canonical_here != 0u ||
         (signed_domain == 0u && inclusive != 0u)) &&
        rund_scan_overflow_)MSL";
  source += type;
  source += R"MSL((running, value, next, signed_domain)) {
      bad = 1u;
    }
    running = next;
  }
  if (bad != 0u) {
    atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
  }
  if (tid == 0u) {
    totals[block] = total;
  }
  if (tid == 0u && block == 0u && resident_count > element_count) {
    atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
  }
}
)MSL";
  return source.valid();
}

} // namespace rund::node::accel::detail
