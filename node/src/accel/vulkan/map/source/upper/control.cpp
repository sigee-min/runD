#include "../upper.hpp"

#include "../../../../kernel/backend/source/storage.hpp"

#include <kernel/program/compute/lowering/vulkan/shape.hpp>

#include <string>
#include <string_view>

namespace rund::node::accel::detail {

namespace vulkan_map_control_source_detail {

inline constexpr std::string_view Prefix = R"glsl(#version 450
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
  args[base + 0u] = dispatch_count == 0u
                        ? 0u
                        : 1u + (dispatch_count - 1u) / )glsl";
inline constexpr std::string_view Suffix = R"glsl(u;
  args[base + 1u] = 1u;
  args[base + 2u] = 1u;
  args[base + 3u] = dispatch_count;
}
)glsl";

} // namespace vulkan_map_control_source_detail

struct VulkanMapControlSourceRecipe final {
  template <typename Sink>
  [[nodiscard]] bool operator()(Sink &sink) const
      noexcept(noexcept(sink.append(std::string_view{}))) {
    return sink.append(vulkan_map_control_source_detail::Prefix) &&
           backend_source_recipe::append_decimal(
               sink, rund::kernel::compute_lowering_detail::kVulkanMapWidth) &&
           sink.append(vulkan_map_control_source_detail::Suffix);
  }
};

[[nodiscard]] bool VulkanMapControlSourceBytes(std::uint64_t &bytes) noexcept {
  return backend_source_recipe::bytes(VulkanMapControlSourceRecipe{}, bytes);
}

std::string VulkanMapControlSource() {
  std::uint64_t bytes = 0u;
  if (!VulkanMapControlSourceBytes(bytes)) {
    return {};
  }
  return backend_source_recipe::materialize(VulkanMapControlSourceRecipe{},
                                            bytes);
}

} // namespace rund::node::accel::detail
