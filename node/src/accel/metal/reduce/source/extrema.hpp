#pragma once

#include "op.hpp"

namespace rund::node::accel::detail {

template <bool Wide, typename Sink>
void AppendMetalReduceExtrema(Sink &source, const rund::kernel::ReduceOp op,
                              const bool signed_domain) {
  const bool minimum = op == rund::kernel::ReduceOp::Min;
  const char *const operation = minimum ? "min" : "max";
  const char *const type = Wide ? "ulong" : "uint";
  const char *const suffix = Wide ? "u64" : "u32";
  const char *const ordered_type =
      signed_domain ? (Wide ? "long" : "int") : type;
  const char *const identity =
      Wide ? (minimum ? (signed_domain ? "0x7ffffffffffffffful"
                                       : "0xfffffffffffffffful")
                      : (signed_domain ? "0x8000000000000000ul" : "0ul"))
           : (minimum ? (signed_domain ? "0x7fffffffu" : "0xffffffffu")
                      : (signed_domain ? "0x80000000u" : "0u"));
  source += "inline ";
  source += type;
  source += " rund_extrema_";
  source += suffix;
  source += "(";
  source += type;
  source += " value) {\n";
  if constexpr (Wide) {
    // Lexicographic (ordered high word, low word) is exact integer order.
    // Only lanes with the winning high word may contribute a low word.
    source += "  const uint high = uint(value >> 32u) ^ ";
    source += signed_domain ? "0x80000000u;\n" : "0u;\n";
    source += "  const uint best = simd_";
    source += operation;
    source += "(high);\n  const uint low = simd_";
    source += operation;
    source += "(high == best ? uint(value) : ";
    source += minimum ? "0xffffffffu);\n" : "0u);\n";
    source += "  return (ulong(best ^ ";
    source += signed_domain ? "0x80000000u" : "0u";
    source += ") << 32u) | ulong(low);\n";
  } else {
    source += "  return as_type<uint>(simd_";
    source += operation;
    source += "(as_type<";
    source += ordered_type;
    source += ">(value)));\n";
  }
  source += "}\nkernel void rund_compute_reduce_";
  source += operation;
  source += "_";
  source += suffix;
  source += "(device const ";
  source += type;
  source += "* input [[buffer(0)]], device ";
  source += type;
  source += "* partial [[buffer(1)]], device ";
  source += type;
  source += R"MSL(* output [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]],
    constant ReduceParams& params [[buffer(4)]],
    device const uint* logical_count [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_groups [[simdgroups_per_threadgroup]]) {
  const uint lane = simd_prefix_exclusive_sum(1u);
  const uint lanes = simd_sum(1u);
  const ulong logical_group = ulong(group) * ulong(simd_groups) + ulong(simd_group);
  if (logical_group >= params.grid_size) { return; }
  const ulong begin = logical_group * ulong(RUND_REDUCE_BLOCK_SIZE);
  const ulong resident_count = params.count_words == 2u
      ? (ulong(logical_count[1]) << 32u) | ulong(logical_count[0])
      : (params.count_words == 1u ? ulong(logical_count[0]) : params.input_count);
  const ulong active_count = params.initial_pass != 0u
      ? min(resident_count, params.input_count) : params.input_count;
  if (params.initial_pass != 0u && lane == 0u && logical_group == 0ul) {
    if (resident_count > params.input_count) {
      atomic_fetch_or_explicit(&status[0], 2u, memory_order_relaxed);
    }
    if (active_count == 0ul) {
      atomic_fetch_or_explicit(&status[0], 4u, memory_order_relaxed);
    }
  }
  const ulong end = min(begin + ulong(RUND_REDUCE_BLOCK_SIZE), active_count);
  )MSL";
  source += type;
  source += " value = ";
  source += identity;
  source += R"MSL(;
  for (ulong index = begin + ulong(lane); index < end; index += ulong(lanes)) {
    value = as_type<)MSL";
  source += type;
  source += ">(";
  source += operation;
  source += "(as_type<";
  source += ordered_type;
  source += ">(value), as_type<";
  source += ordered_type;
  source += ">(input[params.input_offset + index])));\n  }\n";
  source += "  value = rund_extrema_";
  source += suffix;
  source += R"MSL((value);
  if (lane == 0u) {
    if (params.final_pass != 0u) { output[0] = value; }
    else { partial[params.output_offset + logical_group] = value; }
  }
}
)MSL";
}

} // namespace rund::node::accel::detail
