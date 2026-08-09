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
MetalRangeDirectUpdate(const RangeOp op) noexcept {
  if (op == RangeOp::Minimum) {
    return "    value = min(value, min(input[left], input[right]));\n";
  }
  if (op == RangeOp::Maximum) {
    return "    value = max(value, max(input[left], input[right]));\n";
  }
  return "    value += input[left] + input[right];\n";
}

[[nodiscard]] inline const char *
MetalRangeSharedUpdate(const RangeOp op) noexcept {
  if (op == RangeOp::Minimum) {
    return "      value = min(value, min(tile[center - step], "
           "tile[center + step]));\n";
  }
  if (op == RangeOp::Maximum) {
    return "      value = max(value, max(tile[center - step], "
           "tile[center + step]));\n";
  }
  return "      value += tile[center - step] + tile[center + step];\n";
}

template <typename Sink>
inline void AppendMetalRangeKernel(Sink &source, const RangeOp op,
                                   const RangeGpuShape shape,
                                   const char *const type,
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
        R"MSL(  const ulong active_lanes = group_base >= params.element_count
                                 ? 0ul
                                 : min(params.element_count - group_base, )MSL";
    (void)source.decimal(shape.width());
    source += R"MSL(ul);
  const ulong group_end = group_base + active_lanes;
    const uint left_inputs = uint(min(group_base, params.radius));
    const uint right_inputs = group_end >= params.element_count
                                  ? 0u
                                  : uint(min(params.element_count - group_end,
                                             params.radius));
    if (ulong(tid) < active_lanes) {
      const )MSL";
    source += type;
    source += R"MSL( center_value = input[group_base + ulong(tid)];
      tile[)MSL";
    (void)source.decimal(shape.shared_radius_capacity());
    source += R"MSL(u + tid] = center_value;
      if (left_inputs == 0u && tid == 0u) {
        for (uint slot = 0u; ulong(slot) < params.radius; ++slot) {
          tile[)MSL";
    (void)source.decimal(shape.shared_radius_capacity());
    source += R"MSL(u - uint(params.radius) + slot] = center_value;
        }
      }
      if (right_inputs == 0u && ulong(tid) + 1ul == active_lanes) {
        for (uint slot = 0u; ulong(slot) < params.radius; ++slot) {
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
      if (tid == 0u && ulong(left_inputs) < params.radius) {
        for (uint slot = 0u;
             ulong(slot) < params.radius - ulong(left_inputs); ++slot) {
          tile[)MSL";
    (void)source.decimal(shape.shared_radius_capacity());
    source += R"MSL(u - uint(params.radius) + slot] = left_value;
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
      if (tid + 1u == right_inputs && ulong(right_inputs) < params.radius) {
        for (uint slot = right_inputs; ulong(slot) < params.radius; ++slot) {
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
    )MSL";
    source += type;
    source += R"MSL( value = tile[center];
    for (uint step = 1u; ulong(step) <= params.radius; ++step) {
)MSL";
    source += MetalRangeSharedUpdate(op);
    source += R"MSL(    }
    output[i] = value;
  }
)MSL";
    return;
  }
  source += R"MSL(  const ulong i = group_base + ulong(tid);
  if (i >= params.element_count) { return; }
  )MSL";
  source += type;
  source += R"MSL( value = input[i];
  for (ulong step = 1ul; step <= params.radius; ++step) {
    const ulong left = i < step ? 0ul : i - step;
    const ulong right =
        i + step >= params.element_count ? params.element_count - 1ul
                                         : i + step;
)MSL";
  source += MetalRangeDirectUpdate(op);
  source += R"MSL(  }
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
  if (i >= params.element_count) { return; }
  const ulong left = i < params.radius ? 0ul : i - params.radius;
  const ulong right = min(params.element_count - 1ul, i + params.radius);
  )MSL";
  source += type;
  source += R"MSL( value = scratch0[right];
  if (left != 0ul) { value -= scratch0[left - 1ul]; }
  if (i < params.radius) { value += )MSL";
  source += type;
  source += R"MSL((params.radius - i) * input[0]; }
  if (i + params.radius >= params.element_count) {
    value += )MSL";
  source += type;
  source += R"MSL((i + params.radius - (params.element_count - 1ul)) *
             input[params.element_count - 1ul];
  }
  output[i] = value;
}
)MSL";
}

template <typename Sink>
inline void AppendMetalBlockPrefixSuffixKernel(Sink &source, const RangeOp op,
                                               const RangeGpuShape shape,
                                               const char *const type,
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
    const ulong window = params.radius * 2ul + 1ul;
    const ulong begin = base * window;
    const ulong end = min(begin + window, params.stage_element_count);
    for (ulong index = begin; index < end; ++index) {
      const )MSL";
  source += type;
  source += R"MSL( value = index < params.radius
          ? input[0]
          : (index - params.radius < params.element_count
                 ? input[index - params.radius]
                 : input[params.element_count - 1ul]);
      if (index == begin) { forward_values[index] = value; }
      else { forward_values[index] = )MSL";
  source += op == RangeOp::Minimum ? "min" : "max";
  source += R"MSL((forward_values[index - 1ul], value); }
    }
    for (ulong cursor = end; cursor > begin;) {
      const ulong index = cursor - 1ul;
      const )MSL";
  source += type;
  source += R"MSL( value = index < params.radius
          ? input[0]
          : (index - params.radius < params.element_count
                 ? input[index - params.radius]
                 : input[params.element_count - 1ul]);
      if (index + 1ul == end) { backward_values[index] = value; }
      else { backward_values[index] = )MSL";
  source += op == RangeOp::Minimum ? "min" : "max";
  source += R"MSL((value, backward_values[index + 1ul]); }
      cursor = index;
    }
    return;
  }
  const ulong i = base;
  if (i >= params.element_count) { return; }
  const ulong right = i + params.radius * 2ul;
  output[i] = )MSL";
  source += op == RangeOp::Minimum ? "min" : "max";
  source += R"MSL((backward_values[i], forward_values[right]);
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
  ulong element_count;
  ulong radius;
  ulong stage_element_count;
  ulong stage_aux_count;
  uint stage;
  uint reserved;
};

)MSL";
  if (candidate == RangePath::PrefixDifference) {
    if (op != RangeOp::Sum) {
      return false;
    }
    AppendMetalPrefixDifferenceKernel(source, shape, "uint", "u32");
    AppendMetalPrefixDifferenceKernel(source, shape, "ulong", "u64");
    AppendMetalPrefixDifferenceKernel(source, shape, "uint", "i32");
    AppendMetalPrefixDifferenceKernel(source, shape, "ulong", "i64");
  } else if (candidate == RangePath::BlockPrefixSuffix) {
    if (op == RangeOp::Sum) {
      return false;
    }
    AppendMetalBlockPrefixSuffixKernel(source, op, shape, "uint", "u32");
    AppendMetalBlockPrefixSuffixKernel(source, op, shape, "ulong", "u64");
    AppendMetalBlockPrefixSuffixKernel(source, op, shape, "int", "i32");
    AppendMetalBlockPrefixSuffixKernel(source, op, shape, "long", "i64");
  } else {
    AppendMetalRangeKernel(source, op, shape, "uint", "u32");
    AppendMetalRangeKernel(source, op, shape, "ulong", "u64");
    AppendMetalRangeKernel(source, op, shape,
                           op == RangeOp::Sum ? "uint" : "int", "i32");
    AppendMetalRangeKernel(source, op, shape,
                           op == RangeOp::Sum ? "ulong" : "long", "i64");
  }
  return source.valid();
}

} // namespace rund::node::accel::detail
