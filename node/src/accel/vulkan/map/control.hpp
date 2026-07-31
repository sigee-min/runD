#pragma once

#include "../../kernel/backend/run.hpp"
#include "../buffer/create/telemetry.hpp"
#include "../collective/pipeline.hpp"
#include "../descriptor.hpp"
#include "local.hpp"

#include <kernel/program/compute/lowering/vulkan/shape.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct VulkanMapControlPush final {
  std::array<std::uint32_t, 16u> words{};
};

static_assert(sizeof(VulkanMapControlPush) == 64u);

[[nodiscard]] inline bool ReplaceControl(std::string &source,
                                         const std::string &needle,
                                         const std::string &replacement) {
  const std::size_t at = source.find(needle);
  if (needle.empty() || at == std::string::npos ||
      source.find(needle, at + needle.size()) != std::string::npos) {
    return false;
  }
  source.replace(at, needle.size(), replacement);
  return true;
}

[[nodiscard]] inline std::string VulkanMapControlSource() {
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

[[nodiscard]] inline std::pair<std::uint64_t, std::uint64_t>
VulkanMapCheckHash(const VulkanMapEncodeResources &resources) noexcept {
  std::uint64_t hi = resources.plan.op_hash_hi ^ 0x6d61702e63686563ull;
  std::uint64_t lo = resources.plan.op_hash_lo ^ 0x6b2e696e64657800ull;
  const auto mix = [](std::uint64_t &hash, const std::uint64_t value) {
    hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
  };
  for (const VulkanMapCheck check : resources.checks) {
    mix(hi, check.binding);
    mix(hi, check.limit);
    mix(lo, check.offset);
    mix(lo, check.stride);
  }
  return {hi, lo};
}

[[nodiscard]] inline rund::kernel::LoweringArtifact
VulkanMapCheckArtifact(const VulkanMapEncodeResources &resources) {
  std::string source = R"glsl(#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer CountSource { uint count_words[]; };
layout(set = 0, binding = 1, std430) readonly buffer PredicateSource { uint predicate_words[]; };
)glsl";
  for (std::size_t index = 0u; index < resources.checks.size(); ++index) {
    source += "layout(set = 0, binding = " + std::to_string(index + 2u) +
              ", std430) readonly buffer Index" + std::to_string(index) +
              " { uint index" + std::to_string(index) + "_words[]; };\n";
  }
  const std::size_t status_binding = resources.checks.size() + 2u;
  source += "layout(set = 0, binding = " + std::to_string(status_binding) +
            ", std430) buffer ControlStatus { uint status[]; };\n";
  source += R"glsl(
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
  for (std::size_t index = 0u; index < resources.checks.size(); ++index) {
    const VulkanMapCheck check = resources.checks[index];
    source += "      if (uint64_t(index" + std::to_string(index) +
              "_words[uint((" + std::to_string(check.offset) +
              "ul + ordinal * " + std::to_string(check.stride) +
              "ul) / 4ul)]) >= " + std::to_string(check.limit) +
              "ul) { local_invalid = min(local_invalid, "
              "uint(min(ordinal, uint64_t(0xfffffffeu)))); }\n";
  }
  source += R"glsl(    }
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
  const auto [hi, lo] = VulkanMapCheckHash(resources);
  rund::kernel::LoweringArtifact artifact{};
  artifact.key.api = rund::kernel::ComputeApi::Vulkan;
  artifact.key.scalar = resources.plan.scalar;
  artifact.key.domain = resources.plan.domain;
  artifact.key.fixed_format = resources.plan.fixed_format;
  artifact.key.op_hash_hi = hi;
  artifact.key.op_hash_lo = lo;
  artifact.key.canonical_ir_hash_hi = hi;
  artifact.key.canonical_ir_hash_lo = lo;
  artifact.kind = rund::kernel::LoweringArtifactKind::VulkanSource;
  artifact.source_text = std::move(source);
  artifact.ok = true;
  artifact.reason = "ok";
  return artifact;
}

[[nodiscard]] inline rund::kernel::LoweringArtifact
VulkanMapControlArtifact(const rund::kernel::ComputePlan &plan) {
  rund::kernel::LoweringArtifact artifact{};
  artifact.key.api = rund::kernel::ComputeApi::Vulkan;
  artifact.key.scalar = plan.scalar;
  artifact.key.domain = plan.domain;
  artifact.key.fixed_format = plan.fixed_format;
  artifact.kind = rund::kernel::LoweringArtifactKind::VulkanSource;
  artifact.source_text = VulkanMapControlSource();
  artifact.ok = true;
  artifact.reason = "ok";
  return artifact;
}

[[nodiscard]] inline rund::kernel::LoweringArtifact
VulkanControlledMapArtifact(const rund::kernel::LoweringArtifact &source,
                            const rund::kernel::ComputePlan &plan) {
  rund::kernel::LoweringArtifact artifact = source;
  const std::uint64_t binding =
      plan.input_buffer_count + plan.output_buffer_count + 1u;
  const std::string entry = "void main() {\n";
  const std::string guard =
      "  if (gid >= rund_dispatch.tile_count) { return; }\n";
  const std::string controlled_guard =
      "  if (gid >= rund_control_args[rund_dispatch.tile_count * 4u + 3u]) "
      "{ return; }\n";
  const std::string declaration =
      "layout(set = 0, binding = " + std::to_string(binding) +
      ", std430) readonly buffer RundControlArgs { uint "
      "rund_control_args[]; };\n";
  if (!ReplaceControl(artifact.source_text, entry, declaration + entry) ||
      !ReplaceControl(artifact.source_text, guard, controlled_guard) ||
      !ReplaceControl(artifact.source_text, "// artifact_variant=canonical",
                      "// artifact_variant=controlled")) {
    artifact.ok = false;
    artifact.reason = "compute_artifact_mismatch";
    return artifact;
  }
  artifact.key.variant = rund::kernel::LoweringArtifactVariant::Controlled;
  return artifact;
}

[[nodiscard]] inline bool PrepareVulkanMapControl(
    const rund::AccelDevice &pick, const BoundControl &bound,
    const rund::kernel::ComputePlan &plan,
    const std::vector<rund::kernel::ComputeDispatchWindow> &windows,
    VulkanMapEncodeResources &resources) {
  if (!bound.active() && resources.checks.empty()) {
    return true;
  }
  if (windows.empty() ||
      windows.size() > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  resources.control = bound.control;
  if (bound.control.has_count()) {
    if (bound.count == nullptr || bound.count_handle == nullptr) {
      return false;
    }
    resources.control_count =
        LookupVulkanResidentBuffer(pick, *bound.count, *bound.count_handle);
    if (!resources.control_count.check.ok) {
      return false;
    }
    // Lookup returns the canonical allocation identity.  Control addressing,
    // however, is relative to the bound view and must retain its suballocation
    // offset just like ordinary Map bindings do.
    resources.control_count.ref = *bound.count;
  }
  if (bound.control.has_predicate()) {
    if (bound.predicate == nullptr || bound.predicate_handle == nullptr) {
      return false;
    }
    resources.control_predicate = LookupVulkanResidentBuffer(
        pick, *bound.predicate, *bound.predicate_handle);
    if (!resources.control_predicate.check.ok) {
      return false;
    }
    resources.control_predicate.ref = *bound.predicate;
  }
  const VkDeviceSize args_bytes = windows.size() * 4u * sizeof(std::uint32_t);
  VulkanBuffer args{};
  if (!CreateVulkanBuffer(*resources.adapter, args_bytes,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                              VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          args, nullptr, VulkanMemoryUse::Device)) {
    return false;
  }
  resources.control_args = ScopedBuffer{*resources.adapter, args, args_bytes};
  const rund::kernel::LoweringArtifact control_artifact =
      VulkanMapControlArtifact(plan);
  resources.control_pipeline = AcquireVulkanCollectivePipeline(
      *resources.adapter, 4u, sizeof(VulkanMapControlPush), plan,
      control_artifact);
  if (resources.control_pipeline == nullptr ||
      !AcquireVulkanCollectiveDescriptorSet(*resources.adapter,
                                            *resources.control_pipeline, 4u,
                                            resources.control_descriptor) ||
      !CreateVulkanStatus(*resources.adapter, 2u * sizeof(std::uint32_t),
                          resources.control_status)) {
    return false;
  }
  const VulkanBuffer *const count = bound.control.has_count()
                                        ? resources.control_count.device_buffer
                                        : &resources.control_args.buffer;
  const VulkanBuffer *const predicate =
      bound.control.has_predicate() ? resources.control_predicate.device_buffer
                                    : &resources.control_args.buffer;
  if (count == nullptr || predicate == nullptr) {
    return false;
  }
  const auto control_binding =
      [&](const VulkanResidentBufferResult &resident,
          const std::uint64_t offset,
          const rund::kernel::GraphControlSource source, std::uint64_t &base,
          VulkanStorageBinding &binding) {
        const std::uint64_t width =
            source == rund::kernel::GraphControlSource::U64 ? 8u : 4u;
        const std::uint64_t alignment = resources.adapter->storage_align;
        if (resident.device_buffer == nullptr || alignment == 0u ||
            offset > std::numeric_limits<std::uint64_t>::max() -
                         resident.ref.offset_bytes ||
            resident.ref.offset_bytes + offset >
                std::numeric_limits<std::uint64_t>::max() - width) {
          return false;
        }
        const std::uint64_t absolute = resident.ref.offset_bytes + offset;
        base = absolute - absolute % alignment;
        const std::uint64_t range = absolute + width - base;
        if (base > resident.device_buffer->bytes ||
            range > resident.device_buffer->bytes - base ||
            range > resources.adapter->storage_limit) {
          return false;
        }
        binding = VulkanStorageBinding{resident.device_buffer, base, range};
        return true;
      };
  VulkanStorageBinding count_binding{count, 0u, args_bytes};
  VulkanStorageBinding predicate_binding{predicate, 0u, args_bytes};
  if ((bound.control.has_count() &&
       !control_binding(
           resources.control_count, bound.control.count_byte_offset,
           bound.control.count_source, resources.count_base, count_binding)) ||
      (bound.control.has_predicate() &&
       !control_binding(resources.control_predicate,
                        bound.control.predicate_byte_offset,
                        bound.control.predicate_source,
                        resources.predicate_base, predicate_binding))) {
    return false;
  }
  const std::array<VulkanStorageBinding, 4u> bindings{
      count_binding,
      predicate_binding,
      VulkanStorageBinding{&resources.control_args.buffer, 0u, args_bytes},
      VulkanStorageBinding{&resources.control_status.device, 0u,
                           2u * sizeof(std::uint32_t)},
  };
  if (!WriteVulkanStorageDescriptorSet(
          *resources.adapter, resources.control_descriptor, bindings)) {
    return false;
  }
  if (resources.checks.empty()) {
    return true;
  }
  const std::uint64_t capacity =
      windows.back().begin_sequence + windows.back().tile_count;
  std::vector<VulkanStorageBinding> check_bindings;
  check_bindings.reserve(resources.checks.size() + 3u);
  check_bindings.push_back(count_binding);
  check_bindings.push_back(predicate_binding);
  for (VulkanMapCheck &check : resources.checks) {
    const auto *const ref =
        resources.bindings.resident_inputs.ref(check.binding);
    const VulkanResidentBufferResult &resident =
        resources.resident.input(check.binding);
    const std::uint64_t alignment = resources.adapter->storage_align;
    if (check.limit == 0u || ref == nullptr || ref->element_bytes != 4u ||
        ref->stride_bytes < 4u || (ref->stride_bytes & 3u) != 0u ||
        resident.device_buffer == nullptr || alignment == 0u ||
        capacity == 0u ||
        !rund::kernel::checked::mul(capacity - 1u, ref->stride_bytes) ||
        !rund::kernel::checked::add(
            (capacity - 1u) * ref->stride_bytes, ref->element_bytes)) {
      return false;
    }
    check.base = ref->offset_bytes - ref->offset_bytes % alignment;
    check.offset = ref->offset_bytes - check.base;
    check.stride = ref->stride_bytes;
    const std::uint64_t payload =
        (capacity - 1u) * ref->stride_bytes + ref->element_bytes;
    if (check.offset > std::numeric_limits<std::uint64_t>::max() - payload) {
      return false;
    }
    const std::uint64_t range = check.offset + payload;
    if (check.base > resident.device_buffer->bytes ||
        range > resident.device_buffer->bytes - check.base ||
        range > resources.adapter->storage_limit) {
      return false;
    }
    check_bindings.push_back(
        VulkanStorageBinding{resident.device_buffer, check.base, range});
  }
  check_bindings.push_back(
      VulkanStorageBinding{&resources.control_status.device, 0u,
                           2u * sizeof(std::uint32_t)});
  const rund::kernel::LoweringArtifact check_artifact =
      VulkanMapCheckArtifact(resources);
  resources.check_pipeline = AcquireVulkanCollectivePipeline(
      *resources.adapter,
      static_cast<std::uint32_t>(check_bindings.size()),
      sizeof(VulkanMapControlPush), plan, check_artifact);
  return resources.check_pipeline != nullptr &&
         AcquireVulkanCollectiveDescriptorSet(
             *resources.adapter, *resources.check_pipeline,
             static_cast<std::uint32_t>(check_bindings.size()),
             resources.check_descriptor) &&
         WriteVulkanStorageDescriptorSet(*resources.adapter,
                                         resources.check_descriptor,
                                         check_bindings.data(),
                                         static_cast<std::uint32_t>(
                                             check_bindings.size()));
}

[[nodiscard]] inline VulkanMapControlPush
VulkanMapControlParameters(const VulkanMapEncodeResources &map,
                           const rund::kernel::ComputeDispatchWindow &window,
                           const std::uint32_t index) noexcept {
  const auto split = [](const std::uint64_t value) {
    return std::array<std::uint32_t, 2u>{
        static_cast<std::uint32_t>(value),
        static_cast<std::uint32_t>(value >> 32u)};
  };
  const std::uint64_t capacity =
      map.control.capacity == 0u
          ? map.windows.back().begin_sequence + map.windows.back().tile_count
          : map.control.capacity;
  const auto cap = split(capacity);
  const auto expected = split(map.control.predicate_expected);
  const auto begin = split(window.begin_sequence);
  const auto count = split(window.tile_count);
  const std::uint64_t count_offset =
      map.control_count.ref.offset_bytes + map.control.count_byte_offset;
  const std::uint64_t predicate_offset =
      map.control_predicate.ref.offset_bytes +
      map.control.predicate_byte_offset;
  VulkanMapControlPush params{};
  params.words = {
      map.control.has_count() ? 1u : 0u,
      map.control.count_source == rund::kernel::GraphControlSource::U64 ? 1u
                                                                        : 0u,
      map.control.has_predicate() ? 1u : 0u,
      map.control.predicate_source == rund::kernel::GraphControlSource::U64
          ? 1u
          : 0u,
      cap[0],
      cap[1],
      expected[0],
      expected[1],
      begin[0],
      begin[1],
      count[0],
      count[1],
      index,
      static_cast<std::uint32_t>((count_offset - map.count_base) /
                                 sizeof(std::uint32_t)),
      static_cast<std::uint32_t>((predicate_offset - map.predicate_base) /
                                 sizeof(std::uint32_t)),
      map.checks.empty() ? 0u : 1u};
  return params;
}

inline void
EncodeVulkanMapCheck(VkCommandBuffer command,
                     const VulkanMapEncodeResources &map) {
  if (map.checks.empty()) {
    return;
  }
  const VulkanMapControlPush params =
      VulkanMapControlParameters(map, map.windows.front(), 0u);
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     map.check_pipeline->pipeline);
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        map.check_pipeline->pipeline_layout, 0u, 1u,
                        &map.check_descriptor, 0u, nullptr);
  PushVulkanConstants(command, map.check_pipeline->pipeline_layout,
                      VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(params), &params);
  DispatchVulkan(command, 1u, 1u, 1u);
  const VkBufferMemoryBarrier barrier{
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .pNext = nullptr,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = map.control_status.device.buffer,
      .offset = 0u,
      .size = 2u * sizeof(std::uint32_t),
  };
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                       1u, &barrier, 0u, nullptr);
}

inline void
EncodeVulkanMapControl(VkCommandBuffer command,
                       const VulkanMapEncodeResources &map,
                       const rund::kernel::ComputeDispatchWindow &window,
                       const std::uint32_t index) {
  const VulkanMapControlPush params =
      VulkanMapControlParameters(map, window, index);
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     map.control_pipeline->pipeline);
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        map.control_pipeline->pipeline_layout, 0u, 1u,
                        &map.control_descriptor, 0u, nullptr);
  PushVulkanConstants(command, map.control_pipeline->pipeline_layout,
                      VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(params), &params);
  DispatchVulkan(command, 1u, 1u, 1u);
  const VkBufferMemoryBarrier barrier{
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .pNext = nullptr,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = map.control_args.buffer.buffer,
      .offset = index * 4u * sizeof(std::uint32_t),
      .size = 4u * sizeof(std::uint32_t),
  };
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0u, 0u, nullptr, 1u,
                       &barrier, 0u, nullptr);
}

#endif

} // namespace rund::node::accel::detail
