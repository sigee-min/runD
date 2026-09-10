#pragma once

#include "../../../kernel/backend/source/sink.hpp"

namespace rund::node::accel::detail {

template <bool Wide, typename Sink>
[[nodiscard]] bool AppendMetalScanOffsetSource(Sink &sink) noexcept(
    noexcept(sink.append(std::string_view{}))) {
  backend_source_recipe::SourceBuilder<Sink> source{sink};
  constexpr std::string_view type = Wide ? "ulong" : "uint";
  constexpr std::string_view suffix = Wide ? "u64" : "u32";
  source += R"MSL(
kernel void rund_compute_scan_offset_)MSL";
  source += suffix;
  source += R"MSL((
    device )MSL";
  source += type;
  source += R"MSL(* output [[buffer(0)]],
)MSL";
  source += "    device const ";
  source += type;
  source += "* offsets [[buffer(1)]],\n";
  source += R"MSL(    device atomic_uint* status [[buffer(2)]],
    constant ulong& element_count [[buffer(3)]],
    constant ulong& block_size [[buffer(4)]],
    device const uint* logical_count [[buffer(5)]],
    constant uint& count_words [[buffer(6)]],
    constant uint& signed_domain [[buffer(7)]],
    device const )MSL";
  source += type;
  source += R"MSL(* input [[buffer(8)]],
    constant uint& inclusive [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]],
    uint block [[threadgroup_position_in_grid]]) {
  const ulong begin = ulong(block) * block_size;
  const ulong resident_count = count_words == 2u
      ? (ulong(logical_count[1]) << 32u) | ulong(logical_count[0])
      : (count_words == 1u ? ulong(logical_count[0]) : element_count);
  const ulong active_count = min(resident_count, element_count);
  const ulong end = min(begin + block_size, active_count);
  const ulong lane_size =
      (block_size + ulong(width) - 1ul) / ulong(width);
  const ulong lane_begin = min(begin + ulong(tid) * lane_size, end);
  const ulong lane_end = min(lane_begin + lane_size, end);
  const )MSL";
  source += type;
  source += R"MSL( offset = offsets[block];
  uint bad = 0u;
  for (ulong index = lane_begin; index < lane_end; ++index) {
    const )MSL";
  source += type;
  source += R"MSL( local = output[index];
    const )MSL";
  source += type;
  source += R"MSL( global = offset + local;
    if (signed_domain == 0u && inclusive != 0u) {
      // Local unsigned wraps are recorded by block. This tests the carry
      // from the preceding blocks without re-reading the original input.
      if (global < offset) { bad = 1u; }
      output[index] = global;
    } else {
      const )MSL";
  source += type;
  source += R"MSL( value = input[index];
      const )MSL";
  source += type;
  source += R"MSL( previous = global - value;
      if (rund_scan_overflow_)MSL";
  source += type;
  source += R"MSL((previous, value, global, signed_domain)) {
        bad = 1u;
      }
      output[index] = inclusive != 0u ? global : previous;
    }
  }
  if (bad != 0u) {
    atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
  }
}
)MSL";
  return source.valid();
}

} // namespace rund::node::accel::detail
