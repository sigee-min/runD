#pragma once

#include "pipeline/source_artifact.hpp"

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/model.hpp>

#include <cstdint>
#include <string_view>

namespace rund::node::accel::detail {

[[nodiscard]] inline constexpr std::string_view
VulkanResetSourceText() noexcept {
  return R"GLSL(#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = 256) in;
layout(set=0,binding=0,std430) buffer Target { uint target[]; };
layout(push_constant) uniform Params {
  uint64_t count;
  uint64_t base;
  uint64_t offset_words;
  uint64_t stride_words;
  uint element_words;
  uint reserved;
} p;
void main() {
  const uint gid = gl_GlobalInvocationID.x;
  const uint64_t ordinal = p.base + uint64_t(gid);
  if (ordinal >= p.count) { return; }
  const uint64_t word = p.offset_words + ordinal * p.stride_words;
  target[uint(word)] = 0u;
  if (p.element_words == 2u) { target[uint(word + 1ul)] = 0u; }
}
)GLSL";
}

[[nodiscard]] inline rund::kernel::ComputePlan VulkanResetPlan() noexcept {
  return rund::kernel::ComputePlan{
      .op_hash_hi = 0x636f6d707574652eull,
      .op_hash_lo = 0x7265736574000000ull,
      .api = rund::kernel::ComputeApi::Vulkan,
      .scalar = rund::kernel::ComputeScalar::Lane32,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] inline rund::kernel::LoweringArtifact VulkanResetArtifact() {
  rund::kernel::LoweringArtifact artifact =
      VulkanFixedSourceArtifact(VulkanResetSourceText());
  artifact.key.api = rund::kernel::ComputeApi::Vulkan;
  artifact.key.scalar = rund::kernel::ComputeScalar::Lane32;
  artifact.key.op_hash_hi = 0x636f6d707574652eull;
  artifact.key.op_hash_lo = 0x7265736574000000ull;
  return artifact;
}

} // namespace rund::node::accel::detail
