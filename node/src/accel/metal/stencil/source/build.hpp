#pragma once

#include "../../../kernel/backend/source_recipe.hpp"
#include "../local.hpp"

#include <string_view>

namespace rund::node::accel::detail {

[[nodiscard]] inline const char *
MetalStencilSourceOpName(const rund::kernel::StencilOp op) noexcept {
  if (op == rund::kernel::StencilOp::Min) {
    return "min";
  }
  if (op == rund::kernel::StencilOp::Max) {
    return "max";
  }
  return "sum";
}

[[nodiscard]] inline const char *
MetalStencilDirectUpdateLine(const rund::kernel::StencilOp op) noexcept {
  if (op == rund::kernel::StencilOp::Min) {
    return "    value = min(value, min(input[left], input[right]));\n";
  }
  if (op == rund::kernel::StencilOp::Max) {
    return "    value = max(value, max(input[left], input[right]));\n";
  }
  return "    value += input[left] + input[right];\n";
}

[[nodiscard]] inline const char *
MetalStencilSharedUpdateLine(const rund::kernel::StencilOp op) noexcept {
  if (op == rund::kernel::StencilOp::Min) {
    return "      value = min(value, min(tile[center - step], "
           "tile[center + step]));\n";
  }
  if (op == rund::kernel::StencilOp::Max) {
    return "      value = max(value, max(tile[center - step], "
           "tile[center + step]));\n";
  }
  return "      value += tile[center - step] + tile[center + step];\n";
}

template <typename Sink>
inline void
AppendMetalStencilKernel(Sink &source, const rund::kernel::StencilOp op,
                         const char *const type, const char *const suffix) {
  source += "kernel void rund_compute_stencil_";
  source += MetalStencilSourceOpName(op);
  source += "_";
  source += suffix;
  source += R"MSL((
    device const )MSL";
  source += type;
  source += R"MSL(* input [[buffer(0)]],
    device )MSL";
  source += type;
  source += R"MSL(* output [[buffer(1)]],
    constant StencilParams& params [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  threadgroup )MSL";
  source += type;
  source += " tile[";
  (void)source.decimal(kStencilSharedElementCapacity);
  source += R"MSL(];
  const ulong group_base = ulong(group) * )MSL";
  (void)source.decimal(kStencilPhysicalGroupWidth);
  source += R"MSL(ul;
  const ulong active_lanes = group_base >= params.element_count
                                 ? 0ul
                                 : min(params.element_count - group_base, )MSL";
  (void)source.decimal(kStencilPhysicalGroupWidth);
  source += R"MSL(ul);
  if (params.radius <= )MSL";
  (void)source.decimal(kStencilSharedRadiusCap);
  source += R"MSL(ul) {
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
  (void)source.decimal(kStencilSharedRadiusCap);
  source += R"MSL(u + tid] = center_value;
      if (left_inputs == 0u && tid == 0u) {
        for (uint slot = 0u; ulong(slot) < params.radius; ++slot) {
          tile[)MSL";
  (void)source.decimal(kStencilSharedRadiusCap);
  source += R"MSL(u - uint(params.radius) + slot] = center_value;
        }
      }
      if (right_inputs == 0u && ulong(tid) + 1ul == active_lanes) {
        for (uint slot = 0u; ulong(slot) < params.radius; ++slot) {
          tile[)MSL";
  (void)source.decimal(kStencilSharedRadiusCap);
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
  (void)source.decimal(kStencilSharedRadiusCap);
  source += R"MSL(u - left_inputs + tid] = left_value;
      if (tid == 0u && ulong(left_inputs) < params.radius) {
        for (uint slot = 0u;
             ulong(slot) < params.radius - ulong(left_inputs); ++slot) {
          tile[)MSL";
  (void)source.decimal(kStencilSharedRadiusCap);
  source += R"MSL(u - uint(params.radius) + slot] = left_value;
        }
      }
    }
    if (tid < right_inputs) {
      const )MSL";
  source += type;
  source += R"MSL( right_value = input[group_end + ulong(tid)];
      tile[)MSL";
  (void)source.decimal(kStencilSharedRadiusCap);
  source += R"MSL(u + uint(active_lanes) + tid] = right_value;
      if (tid + 1u == right_inputs && ulong(right_inputs) < params.radius) {
        for (uint slot = right_inputs; ulong(slot) < params.radius; ++slot) {
          tile[)MSL";
  (void)source.decimal(kStencilSharedRadiusCap);
  source += R"MSL(u + uint(active_lanes) + slot] = right_value;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ulong(tid) >= active_lanes) { return; }
    const ulong i = group_base + ulong(tid);
    const uint center = )MSL";
  (void)source.decimal(kStencilSharedRadiusCap);
  source += R"MSL(u + tid;
    )MSL";
  source += type;
  source += R"MSL( value = tile[center];
    for (uint step = 1u; ulong(step) <= params.radius; ++step) {
)MSL";
  source += MetalStencilSharedUpdateLine(op);
  source += R"MSL(    }
    output[i] = value;
    return;
  }
  const ulong i = group_base + ulong(tid);
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
  source += MetalStencilDirectUpdateLine(op);
  source += R"MSL(  }
  output[i] = value;
}
)MSL";
}

template <typename Sink>
[[nodiscard]] bool
EmitMetalStencilSource(Sink &sink, const rund::kernel::StencilOp op) noexcept(
    noexcept(sink.append(std::string_view{}))) {
  backend_source_recipe::SourceBuilder<Sink> source{sink};
  source += R"MSL(
#include <metal_stdlib>
using namespace metal;

struct StencilParams {
  ulong element_count;
  ulong radius;
};

)MSL";
  AppendMetalStencilKernel(source, op, "uint", "u32");
  AppendMetalStencilKernel(source, op, "ulong", "u64");
  AppendMetalStencilKernel(
      source, op, op == rund::kernel::StencilOp::Sum ? "uint" : "int", "i32");
  AppendMetalStencilKernel(
      source, op, op == rund::kernel::StencilOp::Sum ? "ulong" : "long", "i64");
  return source.valid();
}

} // namespace rund::node::accel::detail
