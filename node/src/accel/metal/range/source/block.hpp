#pragma once

#include "algebra.hpp"

namespace rund::node::accel::detail {

template <typename Sink>
inline void AppendMetalBlockPrefixSuffixKernel(Sink &source, const RangeOp op,
                                               const RangeBoundary boundary,
                                               const RangeExec &shape,
                                               const char *const type,
                                               const char *const suffix,
                                               const char *const identity) {
  const char *const combine = op == RangeOp::Minimum ? "min" : "max";
  const bool wide =
      std::string_view{suffix} == "u64" || std::string_view{suffix} == "i64";
  for (const char *direction : {"up", "down", "broadcast"}) {
    source += "inline ";
    source += type;
    source += " rund_block_";
    source += direction;
    source += "_";
    source += suffix;
    source += "(";
    source += type;
    source += " value, ushort distance) { return ";
    if (wide) {
      source += "as_type<";
      source += type;
      source += std::string_view{direction} == "broadcast"
                    ? ">(ulong(simd_shuffle"
                    : ">(ulong(simd_shuffle_";
      source += std::string_view{direction} == "broadcast" ? "" : direction;
      source += "(uint(as_type<ulong>(value)), distance)) | (ulong(";
      source += std::string_view{direction} == "broadcast" ? "simd_shuffle"
                                                           : "simd_shuffle_";
      source += std::string_view{direction} == "broadcast" ? "" : direction;
      source += "(uint(as_type<ulong>(value) >> 32u), distance)) << 32u))";
    } else {
      source += std::string_view{direction} == "broadcast" ? "simd_shuffle"
                                                           : "simd_shuffle_";
      source += std::string_view{direction} == "broadcast" ? "" : direction;
      source += "(value, distance)";
    }
    source += "; }\n";
  }
  source += "inline ";
  source += type;
  source += " rund_block_load_";
  source += suffix;
  source += "(device const ";
  source += type;
  source += R"MSL(* input, constant RangeParams& params, ulong index) {
  if (index < params.padding) { return )MSL";
  source += boundary == RangeBoundary::Clamp ? "input[0]" : identity;
  source += R"MSL(; }
  const ulong at = index - params.padding;
  return at < params.input_count ? input[at] : )MSL";
  source += boundary == RangeBoundary::Clamp ? "input[params.input_count - 1ul]"
                                             : identity;
  source += R"MSL(;
}
kernel void rund_range_)MSL";
  source += MetalRangeOpName(op);
  source += "_";
  source += suffix;
  source += "(device const ";
  source += type;
  source += "* input [[buffer(0)]], device ";
  source += type;
  source += "* output [[buffer(1)]], constant RangeParams& params "
            "[[buffer(2)]], device ";
  source += type;
  source += R"MSL(* forward_values [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint sw [[threads_per_simdgroup]]) {
  constexpr uint W = )MSL";
  (void)source.decimal(shape.width());
  source += R"MSL(u;
  const bool prepare = params.stage == )MSL";
  (void)source.decimal(
      static_cast<std::uint32_t>(RangeStageKind::BlockPrefixSuffix));
  source += R"MSL(u;
    if (sg != 0u || ulong(group) >= params.stage_aux_count) { return; }
    const ulong block = ulong(group) * params.window_size;
    const ulong count = prepare ? min(params.window_size, params.stage_element_count - block) : params.window_size;
    const uint active = min(W, simd_sum(1u));
    const ulong begin = block, end = block + count;
    )MSL";
  source += type;
  source += " carry = ";
  source += identity;
  source += ";\n";
  for (const bool reverse : {false, true}) {
    source += reverse ? "    } else {\n" : "    if (prepare) {\n";
    source += reverse
                  ? "    for (ulong remaining = end - begin; remaining > 0ul;) "
                    "{\n      const ulong size = min(ulong(active), "
                    "remaining);\n      remaining -= size;\n      const ulong "
                    "at = begin + remaining + lane;\n"
                  : "    for (ulong tile = begin; tile < end; tile += active) "
                    "{\n      const ulong size = min(ulong(active), end - "
                    "tile);\n      const ulong at = tile + lane;\n";
    source += "      ";
    source += type;
    source += " value = ulong(lane) < size ? rund_block_load_";
    source += suffix;
    source += "(input, params, at) : ";
    source += identity;
    source += ";\n";
    source += "      for (ushort distance = 1u; distance < sw; distance *= 2u) "
              "{\n        const ";
    source += type;
    source += " other = rund_block_";
    source += reverse ? "down_" : "up_";
    source += suffix;
    source += "(value, distance);\n        if (";
    source += reverse ? "lane + distance < active" : "lane >= distance";
    source += ") { value = ";
    source += combine;
    source += "(value, other); }\n      }\n";
    if (reverse) {
      if (shape.plan().shape().stride() == 1u) {
        source += R"MSL(      const ulong result = at;
      if (ulong(lane) < size && result < params.output_count) {
        output[result] = )MSL";
      } else {
        source += R"MSL(      const ulong result = at / params.stride;
      if (ulong(lane) < size && result < params.output_count && result * params.stride == at) {
        output[result] = )MSL";
      }
      source += combine;
      source += "(";
      source += combine;
      source +=
          "(value, carry), forward_values[at + params.window_size - 1ul]); }\n";
    } else {
      source += "      if (ulong(lane) < size) { forward_values[at] = ";
      source += combine;
      source += "(value, carry); }\n";
    }
    source += "      const ";
    source += type;
    source += " tile_total = rund_block_";
    source += "broadcast_";
    source += suffix;
    source += reverse ? "(value, 0u);\n      "
                      : "(value, ushort(active - 1u));\n      ";
    source += "carry = ";
    source += combine;
    source += "(carry, tile_total);\n    }\n";
  }
  source += R"MSL(    }
}
)MSL";
}

} // namespace rund::node::accel::detail
