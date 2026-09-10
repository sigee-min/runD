#include "internal.hpp"

#include "../../collective/pipeline.hpp"
#include "../pipeline/source/artifact.hpp"

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/backend.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] constexpr std::string_view PublishSource() noexcept {
  return R"GLSL(#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) readonly buffer Source0 {
  uint source0[];
};
layout(set = 0, binding = 1, std430) readonly buffer Source1 {
  uint source1[];
};
layout(set = 0, binding = 2, std430) readonly buffer Source2 {
  uint source2[];
};
layout(set = 0, binding = 3, std430) buffer Target {
  uint target_words[];
};
layout(set = 0, binding = 4, std430) readonly buffer ControlSummary {
  uint control[];
};
layout(set = 0, binding = 5, std430) readonly buffer States {
  uvec2 states[];
};
layout(set = 0, binding = 6, std430) readonly buffer ResidentCount {
  uint resident_count[];
};
layout(push_constant) uniform PublishParams {
  uint64_t count;
  uint64_t source_offset_words[3];
  uint64_t source_stride_words[3];
  uint64_t target_offset_words;
  uint64_t target_stride_words;
  uint element_words;
  uint declared_step_count;
  uint state;
  uint final;
  uint stop;
  uint maximum;
  uint tile;
  uint outer;
  uint kind;
  uint64_t count_offset_words;
} p;
shared uint allowed;
void main() {
  const uint lane = gl_LocalInvocationID.x;
  if (lane == 0u) {
    const uvec2 state = states[p.state];
    if (p.kind == 1u) {
      const uint64_t base = uint64_t(p.outer) * uint64_t(p.tile);
      allowed = control[1] == 0u && state.y == 0u &&
                        base < min(uint64_t(resident_count[uint(
                                            p.count_offset_words)]),
                                   uint64_t(p.maximum))
                    ? 1u
                    : 0u;
    } else {
      allowed = p.stop == 0u
                    ? (control[1] == 0u && control[2] == 0xffffffffu &&
                       control[3] == p.declared_step_count
                           ? 1u
                           : 0u)
                    : (control[1] == 0u && state.x != p.final ? 1u : 0u);
    }
  }
  barrier();
  if (allowed == 0u) { return; }
  const uint64_t group =
      uint64_t(gl_WorkGroupID.x) +
      uint64_t(gl_WorkGroupID.y) * uint64_t(gl_NumWorkGroups.x);
  const uint64_t index = group * 256ul + uint64_t(lane);
  const uint64_t base = uint64_t(p.outer) * uint64_t(p.tile);
  uint64_t active_count = p.count;
  if (p.kind == 1u && allowed != 0u) {
    active_count = min(min(uint64_t(p.tile), uint64_t(p.maximum) - base),
                       uint64_t(resident_count[uint(p.count_offset_words)]) -
                           base);
  }
  if (index >= active_count) { return; }
  const uint current =
      p.kind == 1u ? 0u : (p.stop == 0u ? p.final : states[p.state].x);
  const uint64_t source =
      p.source_offset_words[current] + index * p.source_stride_words[current];
  const uint64_t target =
      p.target_offset_words +
      (p.kind == 1u ? base + index : index) * p.target_stride_words;
  target_words[uint(target)] =
      current == 1u ? source1[uint(source)]
                    : (current == 2u ? source2[uint(source)]
                                     : source0[uint(source)]);
  if (p.element_words == 2u) {
    target_words[uint(target + 1ul)] =
        current == 1u ? source1[uint(source + 1ul)]
                      : (current == 2u ? source2[uint(source + 1ul)]
                                       : source0[uint(source + 1ul)]);
  }
}
)GLSL";
}

} // namespace

VulkanCollectivePipeline *AcquireVulkanPublishPipeline(
    VulkanAdapter &adapter) {
  const rund::kernel::ComputePlan plan{
      .op_hash_hi = 0x7075626c69736833ull,
      .op_hash_lo = 0x3262697472617738ull,
      .api = rund::kernel::ComputeApi::Vulkan,
      .scalar = rund::kernel::ComputeScalar::Lane32,
      .ok = true,
      .reason = "ok",
  };
  const rund::kernel::LoweringArtifact artifact =
      VulkanFixedSourceArtifact(PublishSource());
  if (!artifact.ok) {
    return nullptr;
  }
  return AcquireVulkanCollectivePipeline(
      adapter, 7u, sizeof(VulkanPipelinePublishParams), plan, artifact);
}

std::string_view VulkanPublishSourceText() noexcept { return PublishSource(); }

#endif

} // namespace rund::node::accel::detail
