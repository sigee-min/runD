#pragma once

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/backend.hpp>

#include <cstdint>
#include <limits>
#include <string_view>

namespace rund::node::accel::detail {

namespace vulkan_controlled_map_source_detail {

inline constexpr std::string_view Entry = "void main() {\n";
inline constexpr std::string_view DeclarationPrefix =
    "layout(set = 0, binding = ";
inline constexpr std::string_view DeclarationSuffix =
    ", std430) readonly buffer RundControlArgs { uint "
    "rund_control_args[]; };\n";
inline constexpr std::string_view Guard =
    "  if (gid >= rund_dispatch.tile_count) { return; }\n";
inline constexpr std::string_view ControlledGuard =
    "  if (gid >= rund_control_args[rund_dispatch.tile_count * 4u + 3u]) "
    "{ return; }\n";
inline constexpr std::string_view CanonicalVariant =
    "// artifact_variant=canonical";
inline constexpr std::string_view ControlledVariant =
    "// artifact_variant=controlled";

} // namespace vulkan_controlled_map_source_detail

[[nodiscard]] inline constexpr std::uint64_t
VulkanDecimalDigitCount(std::uint64_t value) noexcept {
  std::uint64_t digits = 1u;
  while (value >= 10u) {
    value /= 10u;
    ++digits;
  }
  return digits;
}

[[nodiscard]] inline bool
VulkanControlledMapSourceUpperBytes(const rund::kernel::ComputePlan &plan,
                                    const std::uint64_t specialized,
                                    std::uint64_t &upper) noexcept {
  using namespace vulkan_controlled_map_source_detail;
  std::uint64_t binding = 0u;
  if (!rund::kernel::checked::add(plan.input_buffer_count,
                                  plan.output_buffer_count, binding) ||
      !rund::kernel::checked::add(binding, 1u, binding)) {
    return false;
  }
  const std::uint64_t growth =
      DeclarationPrefix.size() + VulkanDecimalDigitCount(binding) +
      DeclarationSuffix.size() + ControlledGuard.size() - Guard.size() +
      ControlledVariant.size() - CanonicalVariant.size();
  return rund::kernel::checked::add(specialized, growth, upper);
}

[[nodiscard]] inline constexpr std::string_view
VulkanMapControlSourceText() noexcept {
  return R"glsl(#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer CountSource { uint count_words[]; };
layout(set = 0, binding = 1, std430) readonly buffer PredicateSource { uint predicate_words[]; };
layout(set = 0, binding = 2, std430) writeonly buffer DispatchArgs { uint args[]; };
layout(set = 0, binding = 3, std430) buffer ControlStatus { uint status[]; };
layout(push_constant) uniform ControlPush { uvec4 row0; uvec4 row1; uvec4 row2; uvec4 row3; } control;

uint64_t pair64(uint low, uint high) {
  return uint64_t(low) | (uint64_t(high) << 32u);
}

void main() {
  uint64_t capacity = pair64(control.row1.x, control.row1.y);
  uint64_t logical = capacity;
  if (control.row0.x != 0u) {
    logical = control.row0.y != 0u
                  ? pair64(count_words[control.row3.y],
                           count_words[control.row3.y + 1u])
                  : uint64_t(count_words[control.row3.y]);
  }
  bool overflow = logical > capacity;
  uint prior = status[0];
  bool enabled = true;
  if (control.row0.z != 0u) {
    uint64_t observed = control.row0.w != 0u
                            ? pair64(predicate_words[control.row3.z],
                                     predicate_words[control.row3.z + 1u])
                            : uint64_t(predicate_words[control.row3.z]);
    enabled = observed == pair64(control.row1.z, control.row1.w);
  }
  uint64_t begin = pair64(control.row2.x, control.row2.y);
  uint64_t count = pair64(control.row2.z, control.row2.w);
  uint64_t remaining = !overflow && logical > begin
                           ? logical - begin
                           : uint64_t(0);
  uint dispatch_count =
      enabled && !overflow && (control.row3.w == 0u || prior == 0u)
          ? uint(min(remaining, count))
          : 0u;
  if (control.row3.w == 0u) {
    status[0] = overflow ? 1u : 0u;
  }
  uint base = control.row3.x * 4u;
  args[base + 0u] = (dispatch_count + 63u) / 64u;
  args[base + 1u] = 1u;
  args[base + 2u] = 1u;
  args[base + 3u] = dispatch_count;
}
)glsl";
}

namespace vulkan_map_source_detail {

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

} // namespace vulkan_map_source_detail

[[nodiscard]] inline bool VulkanMapCheckSourceUpperBytes(
    const std::uint64_t check_count, const std::uint64_t offset_digit_bytes,
    const std::uint64_t stride_digit_bytes,
    const std::uint64_t limit_digit_bytes, std::uint64_t &upper) noexcept {
  using rund::kernel::checked::add;
  using rund::kernel::checked::mul;
  if (check_count > std::numeric_limits<std::uint64_t>::max() - 2u) {
    return false;
  }
  std::uint64_t bytes = vulkan_map_source_detail::CheckPrefix.size();
  std::uint64_t item = 0u;
  if (!mul(check_count,
           vulkan_map_source_detail::BindingPrefix.size() +
               vulkan_map_source_detail::BindingIndex.size() +
               vulkan_map_source_detail::BindingWords.size() +
               vulkan_map_source_detail::BindingSuffix.size() +
               vulkan_map_source_detail::CheckLinePrefix.size() +
               vulkan_map_source_detail::CheckLineOffset.size() +
               vulkan_map_source_detail::CheckLineStride.size() +
               vulkan_map_source_detail::CheckLineLimit.size() +
               vulkan_map_source_detail::CheckLineSuffix.size(),
           item) ||
      !add(bytes, item, bytes) ||
      !add(bytes, vulkan_map_source_detail::BindingPrefix.size(), bytes) ||
      !add(bytes, VulkanDecimalDigitCount(check_count + 2u), bytes) ||
      !add(bytes, vulkan_map_source_detail::StatusMiddle.size(), bytes) ||
      !add(bytes, vulkan_map_source_detail::CheckBody.size(), bytes) ||
      !add(bytes, vulkan_map_source_detail::CheckTail.size(), bytes) ||
      !add(bytes, offset_digit_bytes, bytes) ||
      !add(bytes, stride_digit_bytes, bytes) ||
      !add(bytes, limit_digit_bytes, bytes)) {
    return false;
  }
  for (std::uint64_t index = 0u; index < check_count; ++index) {
    if (!add(bytes, 3u * VulkanDecimalDigitCount(index), bytes) ||
        !add(bytes, VulkanDecimalDigitCount(index + 2u), bytes)) {
      return false;
    }
  }
  upper = bytes;
  return true;
}

[[nodiscard]] inline bool
VulkanMapCheckSourceUpperBytes(const rund::kernel::LoweringArtifact &artifact,
                               std::uint64_t &upper) noexcept {
  std::uint64_t check_count = 0u;
  std::uint64_t limit_digits = 0u;
  for (std::size_t index = 0u; index < artifact.metadata.read_routes.size();
       ++index) {
    bool first = true;
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (artifact.metadata.read_routes[prior].index ==
          artifact.metadata.read_routes[index].index) {
        first = false;
        break;
      }
    }
    if (!first) {
      continue;
    }
    if (!rund::kernel::checked::add(
            limit_digits,
            VulkanDecimalDigitCount(artifact.metadata.read_routes[index].count),
            limit_digits) ||
        !rund::kernel::checked::add(check_count, 1u, check_count)) {
      return false;
    }
  }
  std::uint64_t offset_digits = 0u;
  std::uint64_t stride_digits = 0u;
  return rund::kernel::checked::mul(check_count, 20u, offset_digits) &&
         rund::kernel::checked::mul(check_count, 20u, stride_digits) &&
         VulkanMapCheckSourceUpperBytes(check_count, offset_digits,
                                        stride_digits, limit_digits, upper);
}

} // namespace rund::node::accel::detail
