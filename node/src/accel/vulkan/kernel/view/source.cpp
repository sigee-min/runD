#include "internal.hpp"

#include "../../collective/pipeline.hpp"
#include "../pipeline/source/artifact.hpp"

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/model.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] constexpr std::string_view VulkanViewSource() noexcept {
  return R"GLSL(#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = 256) in;
layout(push_constant) uniform Params {
  uint64_t count;
  uint64_t source_offset_words;
  uint64_t source_stride_words;
  uint64_t target_offset_words;
  uint64_t target_stride_words;
  uint element_words;
  uint reserved;
} params;
layout(set = 0, binding = 0, std430) readonly buffer Source {
  uint source_words[];
};
layout(set = 0, binding = 1, std430) buffer Target {
  uint target_words[];
};
void main() {
  const uint64_t group =
      uint64_t(gl_WorkGroupID.x) +
      uint64_t(gl_WorkGroupID.y) * uint64_t(gl_NumWorkGroups.x);
  const uint64_t index =
      group * uint64_t(gl_WorkGroupSize.x) +
      uint64_t(gl_LocalInvocationID.x);
  if (index >= params.count) { return; }
  const uint64_t source = params.source_offset_words +
                          index * params.source_stride_words;
  const uint64_t target = params.target_offset_words +
                          index * params.target_stride_words;
  target_words[uint(target)] = source_words[uint(source)];
  if (params.element_words == 2u) {
    target_words[uint(target + 1ul)] = source_words[uint(source + 1ul)];
  }
}
)GLSL";
}

} // namespace

VulkanCollectivePipeline *AcquireVulkanViewPipeline(VulkanAdapter &adapter) {
  const rund::kernel::ComputePlan pseudo{
      .op_hash_hi = 0x7069706576696577ull,
      .op_hash_lo = 0x636f707975333200ull,
      .api = rund::kernel::ComputeApi::Vulkan,
      .scalar = rund::kernel::ComputeScalar::Lane32,
      .ok = true,
      .reason = "ok",
  };
  const rund::kernel::LoweringArtifact artifact =
      VulkanFixedSourceArtifact(VulkanViewSource());
  if (!artifact.ok) {
    return nullptr;
  }
  return AcquireVulkanCollectivePipeline(adapter, kVulkanViewDescriptorCount,
                                         sizeof(VulkanViewParams), pseudo,
                                         artifact);
}

std::string_view VulkanViewSourceText() noexcept { return VulkanViewSource(); }

#endif

} // namespace rund::node::accel::detail
