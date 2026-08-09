#pragma once

#include "../../../kernel/backend/source_recipe.hpp"
#include "../local.hpp"

#include <string_view>

namespace rund::node::accel::detail {

[[nodiscard]] inline const char *MetalRangeOpName(const RangeOp op) noexcept {
  if (op == RangeOp::Minimum) {
    return "min";
  }
  if (op == RangeOp::Maximum) {
    return "max";
  }
  return "sum";
}

[[nodiscard]] inline const char *
MetalRangeIdentity(const RangeOp op, const std::string_view suffix) noexcept {
  if (op == RangeOp::Minimum) {
    if (suffix == "u32") {
      return "0xffffffffu";
    }
    if (suffix == "u64") {
      return "0xfffffffffffffffful";
    }
    if (suffix == "i32") {
      return "2147483647";
    }
    return "9223372036854775807l";
  }
  if (suffix == "i32") {
    return "(-2147483647 - 1)";
  }
  if (suffix == "i64") {
    return "(-9223372036854775807l - 1l)";
  }
  return suffix == "u64" ? "0ul" : "0u";
}

[[nodiscard]] inline const char *
MetalRangeUpdate(const RangeOp op, const bool saturating) noexcept {
  if (op == RangeOp::Minimum) {
    return "      value = min(value, sample);\n";
  }
  if (op == RangeOp::Maximum) {
    return "      value = max(value, sample);\n";
  }
  return saturating ? "      value = rund_range_add_sat(value, sample);\n"
                    : "      value += sample;\n";
}

template <typename Sink>
inline void
AppendMetalRangeKernel(Sink &source, const RangeOp op,
                       const RangeBoundary boundary, const bool saturating,
                       const RangeGpuShape shape, const char *const type,
                       const char *const suffix) {
  source += "kernel void rund_range_";
  source += MetalRangeOpName(op);
  source += "_";
  source += suffix;
  source += R"MSL((
    device const )MSL";
  source += type;
  source += R"MSL(* input [[buffer(0)]],
    device )MSL";
  source += type;
  source += R"MSL(* output [[buffer(1)]],
    constant RangeParams& params [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
)MSL";
  if (shape.uses_shared_halo()) {
    source += "  threadgroup ";
    source += type;
    source += " tile[";
    (void)source.decimal(shape.shared_element_capacity());
    source += "];\n";
  }
  source += R"MSL(  const ulong group_base = ulong(group) * )MSL";
  (void)source.decimal(shape.width());
  source += "ul;\n";
  if (shape.uses_shared_halo()) {
    source +=
        R"MSL(  const ulong active_lanes = group_base >= params.input_count
                                 ? 0ul
                                 : min(params.input_count - group_base, )MSL";
    (void)source.decimal(shape.width());
    source += R"MSL(ul);
  const ulong group_end = group_base + active_lanes;
    const uint left_inputs = uint(min(group_base, params.padding));
    const uint right_inputs = group_end >= params.input_count
                                  ? 0u
                                  : uint(min(params.input_count - group_end,
                                             params.padding));
    if (ulong(tid) < active_lanes) {
      const )MSL";
    source += type;
    source += R"MSL( center_value = input[group_base + ulong(tid)];
      tile[)MSL";
    (void)source.decimal(shape.shared_radius_capacity());
    source += R"MSL(u + tid] = center_value;
      if (left_inputs == 0u && tid == 0u) {
        for (uint slot = 0u; ulong(slot) < params.padding; ++slot) {
          tile[)MSL";
    (void)source.decimal(shape.shared_radius_capacity());
    source += R"MSL(u - uint(params.padding) + slot] = center_value;
        }
      }
      if (right_inputs == 0u && ulong(tid) + 1ul == active_lanes) {
        for (uint slot = 0u; ulong(slot) < params.padding; ++slot) {
          tile[)MSL";
    (void)source.decimal(shape.shared_radius_capacity());
    source += R"MSL(u + uint(active_lanes) + slot] = center_value;
        }
      }
    }
    if (tid < left_inputs) {
      const )MSL";
    source += type;
    source +=
        R"MSL( left_value = input[group_base - ulong(left_inputs) + ulong(tid)];
      tile[)MSL";
    (void)source.decimal(shape.shared_radius_capacity());
    source += R"MSL(u - left_inputs + tid] = left_value;
      if (tid == 0u && ulong(left_inputs) < params.padding) {
        for (uint slot = 0u;
             ulong(slot) < params.padding - ulong(left_inputs); ++slot) {
          tile[)MSL";
    (void)source.decimal(shape.shared_radius_capacity());
    source += R"MSL(u - uint(params.padding) + slot] = left_value;
        }
      }
    }
    if (tid < right_inputs) {
      const )MSL";
    source += type;
    source += R"MSL( right_value = input[group_end + ulong(tid)];
      tile[)MSL";
    (void)source.decimal(shape.shared_radius_capacity());
    source += R"MSL(u + uint(active_lanes) + tid] = right_value;
      if (tid + 1u == right_inputs && ulong(right_inputs) < params.padding) {
        for (uint slot = right_inputs; ulong(slot) < params.padding; ++slot) {
          tile[)MSL";
    (void)source.decimal(shape.shared_radius_capacity());
    source += R"MSL(u + uint(active_lanes) + slot] = right_value;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ulong(tid) >= active_lanes) { return; }
    const ulong i = group_base + ulong(tid);
    const uint center = )MSL";
    (void)source.decimal(shape.shared_radius_capacity());
    source += R"MSL(u + tid;
    const uint first = center - uint(params.padding);
    )MSL";
    source += type;
    source += R"MSL( value = tile[first];
    for (ulong slot = 1ul; slot < params.window_size; ++slot) {
      const )MSL";
    source += type;
    source += R"MSL( sample = tile[first + uint(slot)];
)MSL";
    source += MetalRangeUpdate(op, saturating);
    source += R"MSL(    }
    output[i] = value;
  }
)MSL";
    return;
  }
  source += R"MSL(  const ulong i = group_base + ulong(tid);
  if (i >= params.output_count) { return; }
  const ulong anchor = i * params.stride;
  )MSL";
  source += type;
  source += R"MSL( value = )MSL";
  source += type;
  source += R"MSL((0);
  bool seeded = false;
  for (ulong slot = 0ul; slot < params.window_size; ++slot) {
    ulong input_index = 0ul;
    bool valid = true;
    if (slot < params.padding) {
      const ulong delta = params.padding - slot;
      if (anchor < delta) {
)MSL";
  if (boundary == RangeBoundary::Clamp) {
    source += "        input_index = 0ul;\n";
  } else {
    source += "        valid = false;\n";
  }
  source += R"MSL(      } else {
        input_index = anchor - delta;
)MSL";
  if (boundary == RangeBoundary::Clamp) {
    source += R"MSL(        if (input_index >= params.input_count) {
          input_index = params.input_count - 1ul;
        }
)MSL";
  } else {
    source += R"MSL(        if (input_index >= params.input_count) {
          valid = false;
        }
)MSL";
  }
  source += R"MSL(      }
    } else {
      const ulong delta = slot - params.padding;
      if (anchor >= params.input_count ||
          delta >= params.input_count - anchor) {
)MSL";
  if (boundary == RangeBoundary::Clamp) {
    source += "        input_index = params.input_count - 1ul;\n";
  } else {
    source += "        valid = false;\n";
  }
  source += R"MSL(      } else {
        input_index = anchor + delta;
      }
    }
    if (!valid) { continue; }
    const )MSL";
  source += type;
  source += R"MSL( sample = input[input_index];
    if (!seeded) {
      value = sample;
      seeded = true;
    } else {
)MSL";
  source += MetalRangeUpdate(op, saturating);
  source += R"MSL(    }
  }
  output[i] = value;
}
)MSL";
}

[[nodiscard]] constexpr std::uint32_t
MetalRangeStageValue(const RangeStageKind stage) noexcept {
  return static_cast<std::uint32_t>(stage);
}

template <typename Sink>
inline void AppendMetalPrefixDifferenceKernel(Sink &source,
                                              const RangeBoundary boundary,
                                              const RangeGpuShape shape,
                                              const char *const type,
                                              const char *const suffix) {
  source += "kernel void rund_range_sum_";
  source += suffix;
  source += R"MSL((
    device const )MSL";
  source += type;
  source += R"MSL(* input [[buffer(0)]],
    device )MSL";
  source += type;
  source += R"MSL(* output [[buffer(1)]],
    constant RangeParams& params [[buffer(2)]],
    device )MSL";
  source += type;
  source += R"MSL(* scratch0 [[buffer(3)]],
    device )MSL";
  source += type;
  source += R"MSL(* scratch1 [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  threadgroup )MSL";
  source += type;
  source += " scan[";
  (void)source.decimal(shape.width());
  source += R"MSL(];
  const ulong group_base = ulong(group) * )MSL";
  (void)source.decimal(shape.width());
  source += R"MSL(ul;
  if (params.stage == )MSL";
  (void)source.decimal(MetalRangeStageValue(RangeStageKind::PrefixBlock));
  source += R"MSL(u || params.stage == )MSL";
  (void)source.decimal(MetalRangeStageValue(RangeStageKind::PrefixSummary));
  source += R"MSL(u) {
    const ulong i = group_base + ulong(tid);
    const bool active = i < params.stage_element_count;
    const )MSL";
  source += type;
  source += R"MSL( value = active
        ? (params.stage == )MSL";
  (void)source.decimal(MetalRangeStageValue(RangeStageKind::PrefixBlock));
  source += R"MSL(u ? input[i] : scratch0[i])
        : )MSL";
  source += type;
  source += R"MSL((0);
    scan[tid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 1u; offset < )MSL";
  (void)source.decimal(shape.width());
  source += R"MSL(u; offset <<= 1u) {
      const uint tree = (tid + 1u) * offset * 2u - 1u;
      if (tree < )MSL";
  (void)source.decimal(shape.width());
  source += R"MSL(u) { scan[tree] += scan[tree - offset]; }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
      if (params.stage_aux_count > 1ul) { scratch1[group] = scan[)MSL";
  (void)source.decimal(shape.width() - 1u);
  source += R"MSL(]; }
      scan[)MSL";
  (void)source.decimal(shape.width() - 1u);
  source += R"MSL(] = )MSL";
  source += type;
  source += R"MSL((0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = )MSL";
  (void)source.decimal(shape.width() / 2u);
  source += R"MSL(u; offset > 0u; offset >>= 1u) {
      const uint tree = (tid + 1u) * offset * 2u - 1u;
      if (tree < )MSL";
  (void)source.decimal(shape.width());
  source += R"MSL(u) {
        const )MSL";
  source += type;
  source += R"MSL( prior = scan[tree - offset];
        scan[tree - offset] = scan[tree];
        scan[tree] += prior;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active) { scratch0[i] = scan[tid] + value; }
    return;
  }
  if (params.stage == )MSL";
  (void)source.decimal(MetalRangeStageValue(RangeStageKind::PrefixFixup));
  source += R"MSL(u) {
    const ulong i = group_base + ulong(tid);
    if (i < params.stage_element_count && group != 0u) {
      scratch0[i] += scratch1[group - 1u];
    }
    return;
  }
  const ulong i = group_base + ulong(tid);
  if (i >= params.output_count) { return; }
  const ulong anchor = i * params.stride;
  const ulong left = anchor < params.padding ? 0ul : anchor - params.padding;
  const ulong right_width = params.window_size - params.padding;
  const ulong right =
      anchor >= params.input_count
          ? params.input_count - 1ul
          : (right_width >= params.input_count - anchor
                 ? params.input_count - 1ul
                 : anchor + right_width - 1ul);
  )MSL";
  source += type;
  source += R"MSL( value = scratch0[right];
  if (left != 0ul) { value -= scratch0[left - 1ul]; }
  )MSL";
  if (boundary == RangeBoundary::Clamp) {
    source += R"MSL(  const ulong left_missing =
      anchor < params.padding ? params.padding - anchor : 0ul;
  const ulong right_missing =
      anchor >= params.input_count
          ? anchor - params.input_count + right_width
          : (right_width > params.input_count - anchor
                 ? right_width - (params.input_count - anchor)
                 : 0ul);
  if (left_missing != 0ul) { value += )MSL";
    source += type;
    source += R"MSL((left_missing * input[0]); }
  if (right_missing != 0ul) { value += )MSL";
    source += type;
    source += R"MSL((right_missing * input[params.input_count - 1ul]); }
)MSL";
  }
  source += R"MSL(
  output[i] = value;
}
)MSL";
}

template <typename Sink>
inline void AppendMetalBlockPrefixSuffixKernel(Sink &source, const RangeOp op,
                                               const RangeBoundary boundary,
                                               const RangeGpuShape shape,
                                               const char *const type,
                                               const char *const suffix,
                                               const char *const identity) {
  source += "kernel void rund_range_";
  source += MetalRangeOpName(op);
  source += "_";
  source += suffix;
  source += R"MSL((
    device const )MSL";
  source += type;
  source += R"MSL(* input [[buffer(0)]],
    device )MSL";
  source += type;
  source += R"MSL(* output [[buffer(1)]],
    constant RangeParams& params [[buffer(2)]],
    device )MSL";
  source += type;
  source += R"MSL(* forward_values [[buffer(3)]],
    device )MSL";
  source += type;
  source += R"MSL(* backward_values [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  const ulong base = ulong(group) * )MSL";
  (void)source.decimal(shape.width());
  source += R"MSL(ul + ulong(tid);
  if (params.stage == )MSL";
  (void)source.decimal(MetalRangeStageValue(RangeStageKind::BlockPrefixSuffix));
  source += R"MSL(u) {
    if (base >= params.stage_aux_count) { return; }
    const ulong window = params.window_size;
    const ulong begin = base * window;
    const ulong end = min(begin + window, params.stage_element_count);
    for (ulong index = begin; index < end; ++index) {
      const )MSL";
  source += type;
  source += R"MSL( value = index < params.padding
          ? )MSL";
  source += boundary == RangeBoundary::Clamp ? "input[0]" : identity;
  source += R"MSL(
          : (index - params.padding < params.input_count
                 ? input[index - params.padding]
                 : )MSL";
  source += boundary == RangeBoundary::Clamp ? "input[params.input_count - 1ul]"
                                             : identity;
  source += R"MSL();
      if (index == begin) { forward_values[index] = value; }
      else { forward_values[index] = )MSL";
  source += op == RangeOp::Minimum ? "min" : "max";
  source += R"MSL((forward_values[index - 1ul], value); }
    }
    for (ulong cursor = end; cursor > begin;) {
      const ulong index = cursor - 1ul;
      const )MSL";
  source += type;
  source += R"MSL( value = index < params.padding
          ? )MSL";
  source += boundary == RangeBoundary::Clamp ? "input[0]" : identity;
  source += R"MSL(
          : (index - params.padding < params.input_count
                 ? input[index - params.padding]
                 : )MSL";
  source += boundary == RangeBoundary::Clamp ? "input[params.input_count - 1ul]"
                                             : identity;
  source += R"MSL();
      if (index + 1ul == end) { backward_values[index] = value; }
      else { backward_values[index] = )MSL";
  source += op == RangeOp::Minimum ? "min" : "max";
  source += R"MSL((value, backward_values[index + 1ul]); }
      cursor = index;
    }
    return;
  }
  const ulong i = base;
  if (i >= params.output_count) { return; }
  const ulong left = i * params.stride;
  const ulong right = left + params.window_size - 1ul;
  output[i] = )MSL";
  source += op == RangeOp::Minimum ? "min" : "max";
  source += R"MSL((backward_values[left], forward_values[right]);
}
)MSL";
}

template <typename Sink>
[[nodiscard]] bool
EmitMetalRangeSource(Sink &sink, const RangeExec &execution) noexcept(
    noexcept(sink.append(std::string_view{}))) {
  const RangeOp op = execution.operation();
  const RangeGpuShape shape = execution.shape();
  const RangePath candidate = execution.candidate();
  backend_source_recipe::SourceBuilder<Sink> source{sink};
  source += R"MSL(
#include <metal_stdlib>
using namespace metal;

struct RangeParams {
  ulong input_count;
  ulong output_count;
  ulong window_size;
  ulong stride;
  ulong padding;
  ulong stage_element_count;
  ulong stage_aux_count;
  uint stage;
  uint reserved;
};

)MSL";
  if (execution.saturating_sum()) {
    source +=
        R"MSL(inline int rund_range_add_sat(const int left, const int right) {
  if (right > 0 && left > 2147483647 - right) { return 2147483647; }
  if (right < 0 && left < (-2147483647 - 1) - right) {
    return (-2147483647 - 1);
  }
  return left + right;
}

inline long rund_range_add_sat(const long left, const long right) {
  if (right > 0l && left > 9223372036854775807l - right) {
    return 9223372036854775807l;
  }
  if (right < 0l && left < (-9223372036854775807l - 1l) - right) {
    return (-9223372036854775807l - 1l);
  }
  return left + right;
}

)MSL";
  }
  if (candidate == RangePath::PrefixDifference) {
    if (op != RangeOp::Sum) {
      return false;
    }
    const RangeBoundary boundary = execution.plan().shape().boundary();
    AppendMetalPrefixDifferenceKernel(source, boundary, shape, "uint", "u32");
    AppendMetalPrefixDifferenceKernel(source, boundary, shape, "ulong", "u64");
    AppendMetalPrefixDifferenceKernel(source, boundary, shape, "uint", "i32");
    AppendMetalPrefixDifferenceKernel(source, boundary, shape, "ulong", "i64");
  } else if (candidate == RangePath::BlockPrefixSuffix) {
    if (op == RangeOp::Sum) {
      return false;
    }
    const RangeBoundary boundary = execution.plan().shape().boundary();
    AppendMetalBlockPrefixSuffixKernel(source, op, boundary, shape, "uint",
                                       "u32", MetalRangeIdentity(op, "u32"));
    AppendMetalBlockPrefixSuffixKernel(source, op, boundary, shape, "ulong",
                                       "u64", MetalRangeIdentity(op, "u64"));
    AppendMetalBlockPrefixSuffixKernel(source, op, boundary, shape, "int",
                                       "i32", MetalRangeIdentity(op, "i32"));
    AppendMetalBlockPrefixSuffixKernel(source, op, boundary, shape, "long",
                                       "i64", MetalRangeIdentity(op, "i64"));
  } else {
    const RangeBoundary boundary = execution.plan().shape().boundary();
    const bool saturating = execution.saturating_sum();
    AppendMetalRangeKernel(source, op, boundary, saturating, shape,
                           saturating ? "int" : "uint", "u32");
    AppendMetalRangeKernel(source, op, boundary, saturating, shape,
                           saturating ? "long" : "ulong", "u64");
    AppendMetalRangeKernel(source, op, boundary, saturating, shape,
                           op == RangeOp::Sum && !saturating ? "uint" : "int",
                           "i32");
    AppendMetalRangeKernel(source, op, boundary, saturating, shape,
                           op == RangeOp::Sum && !saturating ? "ulong" : "long",
                           "i64");
  }
  return source.valid();
}

} // namespace rund::node::accel::detail
