#pragma once

#include "../local.hpp"

#include <string_view>

namespace rund::node::accel::detail {

[[nodiscard]] inline const char* MetalStencilSourceOpName(
    const rund::kernel::StencilOp op) noexcept {
  if (op == rund::kernel::StencilOp::Min) { return "min"; }
  if (op == rund::kernel::StencilOp::Max) { return "max"; }
  return "sum";
}

[[nodiscard]] inline const char* MetalStencilUpdateLine(
    const rund::kernel::StencilOp op) noexcept {
  if (op == rund::kernel::StencilOp::Min) {
    return "    value = min(value, min(input[left], input[right]));\n";
  }
  if (op == rund::kernel::StencilOp::Max) {
    return "    value = max(value, max(input[left], input[right]));\n";
  }
  return "    value += input[left] + input[right];\n";
}

template <typename Sink>
inline void AppendMetalStencilKernel(Sink &source,
                                     const rund::kernel::StencilOp op,
                                     const char *const type,
                                     const char *const suffix) {
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
    uint gid [[thread_position_in_grid]]) {
  const ulong i = ulong(gid);
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
  source += MetalStencilUpdateLine(op);
  source += R"MSL(  }
  output[i] = value;
}
)MSL";
}

template <typename Sink>
[[nodiscard]] bool EmitMetalStencilSource(
    Sink &source, const rund::kernel::StencilOp op) noexcept(
    noexcept(source += std::string_view{})) {
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
  AppendMetalStencilKernel(source, op,
                           op == rund::kernel::StencilOp::Sum ? "uint" : "int",
                           "i32");
  AppendMetalStencilKernel(
      source, op, op == rund::kernel::StencilOp::Sum ? "ulong" : "long",
      "i64");
  return source.valid();
}

}  // namespace rund::node::accel::detail
