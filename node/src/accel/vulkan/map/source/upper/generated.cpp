#include "../upper.hpp"

#include "../../../../kernel/backend/source/storage.hpp"

#include <string>
#include <string_view>

namespace rund::node::accel::detail {

namespace vulkan_generated_map_control_source_detail {

inline constexpr std::string_view Source = R"glsl(#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer CountSource { uint count_words[]; };
layout(set = 0, binding = 1, std430) readonly buffer PredicateSource { uint predicate_words[]; };
layout(set = 0, binding = 2, std430) writeonly buffer DispatchArgs { uint args[]; };
layout(set = 0, binding = 3, std430) buffer ControlStatus { uint status[]; };
layout(set = 0, binding = 4, std430) buffer AdmissionRow { uint admission_words[]; };
layout(set = 0, binding = 5, std430) readonly buffer PipelineSummary { uint summary_words[]; };
layout(push_constant) uniform GeneratedControlPush {
  uvec4 row0; uvec4 row1; uvec4 row2; uvec4 row3;
  uvec4 identity0; uvec4 identity1; uvec4 identity2; uvec4 identity3;
} control;

uint64_t pair64(uint low, uint high) {
  return uint64_t(low) | (uint64_t(high) << 32u);
}

bool valid_admission_row() {
  bool valid = true;
  for (uint index = 0u; index < 58u; ++index) {
    valid = valid && admission_words[index] == admission_words[58u + index];
  }
  const uint64_t coordinate = pair64(admission_words[8u], admission_words[9u]);
  const uint64_t turn = pair64(admission_words[10u], admission_words[11u]);
  const uint stride = admission_words[21u];
  const uint slot = admission_words[22u];
  valid = valid && stride != 0u && slot < stride;
  valid = valid && turn <= (uint64_t(0xfffffffffffffffful) - uint64_t(slot)) /
                         uint64_t(stride);
  valid = valid && coordinate == turn * uint64_t(stride) + uint64_t(slot);
  valid = valid && pair64(admission_words[0u], admission_words[1u]) ==
                         pair64(control.identity0.x, control.identity0.y);
  valid = valid && pair64(admission_words[2u], admission_words[3u]) ==
                         pair64(control.identity0.z, control.identity0.w);
  valid = valid && pair64(admission_words[4u], admission_words[5u]) ==
                         pair64(control.identity1.x, control.identity1.y);
  valid = valid && pair64(admission_words[6u], admission_words[7u]) ==
                         pair64(control.identity1.z, control.identity1.w);
  valid = valid && admission_words[22u] == control.identity2.x &&
          admission_words[21u] == control.identity2.y;
  const uint64_t expected_control =
      uint64_t(control.identity2.z) +
      turn * uint64_t(control.identity2.w);
  const uint64_t expected_descriptor =
      pair64(control.identity3.x, control.identity3.y) +
      turn * pair64(control.identity3.z, control.identity3.w);
  const bool graph_stage = control.identity3.z == 0u &&
                           control.identity3.w == 0u;
  valid = valid && pair64(admission_words[25u], 0u) == expected_control;
  valid = valid && pair64(admission_words[16u], admission_words[17u]) ==
                         expected_descriptor;
  valid = valid &&
          (graph_stage ||
           (summary_words[0u] <= 0xffffffffu &&
            uint64_t(summary_words[0u]) + uint64_t(control.identity2.w) ==
                uint64_t(admission_words[25u])));
  const uint local_count = admission_words[20u];
  const uint step_count = admission_words[24u];
  valid = valid && local_count != 0u && local_count <= 32u &&
          local_count <= step_count;
  for (uint local = 0u; local < 32u; ++local) {
    if (local >= local_count) { break; }
    valid = valid && admission_words[26u + local] < step_count;
    for (uint prior = 0u; prior < local; ++prior) {
      valid = valid && admission_words[26u + local] !=
                           admission_words[26u + prior];
    }
  }
  return valid;
}

void main() {
  const uint base = control.row3.x * 4u;
  const uint service_reason = summary_words[1u] != 0u
                                  ? summary_words[1u]
                                  : summary_words[23u];
  if (service_reason != 0u) {
    args[base + 0u] = 0u;
    args[base + 1u] = 0u;
    args[base + 2u] = 0u;
    args[base + 3u] = 0u;
    status[0] = 1u;
    status[1] = service_reason;
    admission_words[116u] = 0u;
    admission_words[117u] = service_reason;
    admission_words[118u] = 0u;
    admission_words[119u] = 0u;
    return;
  }
  if (status[0] != 0u) {
    // The status buffer uses private markers (1=range check, 2=control
    // check), but the published gate row carries canonical runD Reasons.
    const uint reason = status[0] == 2u ? 0x6009u : 0x6016u;
    args[base + 0u] = 0u;
    args[base + 1u] = 0u;
    args[base + 2u] = 0u;
    args[base + 3u] = 0u;
    admission_words[116u] = 0u;
    admission_words[117u] = reason;
    admission_words[118u] = 0u;
    admission_words[119u] = 0u;
    return;
  }
  if (!valid_admission_row()) {
    args[base + 0u] = 0u;
    args[base + 1u] = 0u;
    args[base + 2u] = 0u;
    args[base + 3u] = 0u;
    status[0] = 2u;
    status[1] = 0xffffffffu;
    admission_words[116u] = 0u;
    admission_words[117u] = 0xffffffffu;
    admission_words[118u] = 0u;
    admission_words[119u] = 0u;
    return;
  }
  const uint64_t capacity = pair64(control.row1.x, control.row1.y);
  uint64_t logical = capacity;
  if (control.row0.x != 0u) {
    logical = control.row0.y != 0u
                  ? pair64(count_words[control.row3.y],
                           count_words[control.row3.y + 1u])
                  : uint64_t(count_words[control.row3.y]);
  }
  if (logical > capacity) {
    args[base + 0u] = 0u;
    args[base + 1u] = 0u;
    args[base + 2u] = 0u;
    args[base + 3u] = 0u;
    status[0] = 1u;
    status[1] = 0x6009u;
    admission_words[116u] = 0u;
    admission_words[117u] = 0x6009u;
    admission_words[118u] = 0u;
    admission_words[119u] = 0u;
    return;
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
  admission_words[116u] = 1u;
  admission_words[117u] = 0u;
  admission_words[118u] = admission_words[16u];
  admission_words[119u] = admission_words[17u];
  const uint64_t begin = pair64(control.row2.x, control.row2.y);
  const uint64_t count = pair64(control.row2.z, control.row2.w);
  const uint64_t remaining = enabled && logical > begin ? logical - begin
                                                          : uint64_t(0);
  const uint dispatch_count =
      remaining < count ? uint(remaining) : uint(count);
  args[base + 0u] = dispatch_count == 0u
                        ? 0u
                        : 1u + (dispatch_count - 1u) / 1u;
  args[base + 1u] = 1u;
  args[base + 2u] = 1u;
  args[base + 3u] = dispatch_count;
}
)glsl";

} // namespace vulkan_generated_map_control_source_detail

struct VulkanGeneratedMapControlSourceRecipe final {
  template <typename Sink>
  [[nodiscard]] bool operator()(Sink &sink) const
      noexcept(noexcept(sink.append(std::string_view{}))) {
    return sink.append(vulkan_generated_map_control_source_detail::Source);
  }
};

[[nodiscard]] bool
VulkanGeneratedMapControlSourceBytes(std::uint64_t &bytes) noexcept {
  return backend_source_recipe::bytes(VulkanGeneratedMapControlSourceRecipe{},
                                      bytes);
}

std::string VulkanGeneratedMapControlSource() {
  std::uint64_t bytes = 0u;
  if (!VulkanGeneratedMapControlSourceBytes(bytes)) {
    return {};
  }
  return backend_source_recipe::materialize(
      VulkanGeneratedMapControlSourceRecipe{}, bytes);
}

} // namespace rund::node::accel::detail
