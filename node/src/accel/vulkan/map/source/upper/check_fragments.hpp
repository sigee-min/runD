#pragma once

#include <string_view>

namespace rund::node::accel::detail::vulkan_map_source_detail {

inline constexpr std::string_view CheckPrefix = R"glsl(#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer CountSource { uint count_words[]; };
layout(set = 0, binding = 1, std430) readonly buffer PredicateSource { uint predicate_words[]; };
)glsl";
inline constexpr std::string_view BindingPrefix = "layout(set = 0, binding = ";
inline constexpr std::string_view BindingIndex =
    ", std430) readonly buffer Index";
inline constexpr std::string_view BindingWords = " { uint index";
inline constexpr std::string_view BindingSuffix = "_words[]; };\n";
inline constexpr std::string_view StatusMiddle =
    ", std430) buffer ControlStatus { uint status[]; };\n";
inline constexpr std::string_view CheckBody = R"glsl(
layout(push_constant) uniform ControlPush { uvec4 row0; uvec4 row1; uvec4 row2; uvec4 row3; } control;
shared uint invalids[256];

uint64_t pair64(uint low, uint high) {
  return uint64_t(low) | (uint64_t(high) << 32u);
}

void main() {
  const uint tid = gl_LocalInvocationID.x;
  const uint64_t capacity = pair64(control.row1.x, control.row1.y);
  uint64_t logical = capacity;
  if (control.row0.x != 0u) {
    logical = control.row0.y != 0u
                  ? pair64(count_words[control.row3.y],
                           count_words[control.row3.y + 1u])
                  : uint64_t(count_words[control.row3.y]);
  }
  bool enabled = true;
  if (control.row0.z != 0u) {
    const uint64_t observed =
        control.row0.w != 0u
            ? pair64(predicate_words[control.row3.z],
                     predicate_words[control.row3.z + 1u])
            : uint64_t(predicate_words[control.row3.z]);
    enabled = observed == pair64(control.row1.z, control.row1.w);
  }
  uint local_invalid = 0xffffffffu;
  if (enabled && logical <= capacity) {
    for (uint64_t ordinal = uint64_t(tid); ordinal < logical;
         ordinal += uint64_t(256)) {
)glsl";
inline constexpr std::string_view CheckLinePrefix = "      if (uint64_t(index";
inline constexpr std::string_view CheckLineOffset = "_words[uint((";
inline constexpr std::string_view CheckLineStride = "ul + ordinal * ";
inline constexpr std::string_view CheckLineLimit = "ul) / 4ul)]) >= ";
inline constexpr std::string_view CheckLineSuffix =
    "ul) { local_invalid = min(local_invalid, "
    "uint(min(ordinal, uint64_t(0xfffffffeu)))); }\n";
inline constexpr std::string_view CheckTail = R"glsl(    }
  }
  invalids[tid] = local_invalid;
  barrier();
  for (uint stride = 128u; stride != 0u; stride >>= 1u) {
    if (tid < stride) {
      invalids[tid] = min(invalids[tid], invalids[tid + stride]);
    }
    barrier();
  }
  if (tid != 0u) { return; }
  status[1] = uint(min(logical, uint64_t(0xffffffffu)));
  status[0] = logical > capacity
                  ? 1u
                  : (invalids[0] == 0xffffffffu ? 0u : 2u);
  if (status[0] == 2u) { status[1] = invalids[0]; }
}
)glsl";

} // namespace rund::node::accel::detail::vulkan_map_source_detail
