#pragma once

#include "../../kernel/backend/exception.hpp"
#include "../../kernel/backend/run.hpp"
#include "../../kernel/backend/source_recipe.hpp"
#include "../buffer/create/telemetry.hpp"
#include "../collective/pipeline.hpp"
#include "../descriptor.hpp"
#include "local.hpp"
#include "source_upper.hpp"

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

[[nodiscard]] inline std::string VulkanMapControlSource() {
  const auto recipe = []<typename Sink>(Sink &sink) noexcept(
                          noexcept(sink.append(std::string_view{}))) {
    return sink.append(VulkanMapControlSourceText());
  };
  return backend_source_recipe::materialize(
      recipe, VulkanMapControlSourceText().size());
}

[[nodiscard]] inline std::pair<std::uint64_t, std::uint64_t>
VulkanMapCheckHash(const VulkanMapTemplateResources &prepared) noexcept {
  std::uint64_t hi = prepared.plan.op_hash_hi ^ 0x6d61702e63686563ull;
  std::uint64_t lo = prepared.plan.op_hash_lo ^ 0x6b2e696e64657800ull;
  const auto mix = [](std::uint64_t &hash, const std::uint64_t value) {
    hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
  };
  for (const VulkanMapCheck check : prepared.checks) {
    mix(hi, check.binding);
    mix(hi, check.limit);
    mix(lo, check.offset);
    mix(lo, check.stride);
  }
  return {hi, lo};
}

struct VulkanMapCheckSourceRecipe final {
  const VulkanMapTemplateResources &prepared;

  template <typename Sink>
  [[nodiscard]] bool operator()(Sink &sink) const
      noexcept(noexcept(sink.append(std::string_view{}))) {
    using namespace vulkan_map_source_detail;
    if (!sink.append(CheckPrefix)) {
      return false;
    }
    for (std::size_t index = 0u; index < prepared.checks.size(); ++index) {
      if (!sink.append(BindingPrefix) ||
          !backend_source_recipe::append_decimal(sink, index + 2u) ||
          !sink.append(BindingIndex) ||
          !backend_source_recipe::append_decimal(sink, index) ||
          !sink.append(BindingWords) ||
          !backend_source_recipe::append_decimal(sink, index) ||
          !sink.append(BindingSuffix)) {
        return false;
      }
    }
    if (!sink.append(BindingPrefix) ||
        !backend_source_recipe::append_decimal(sink,
                                               prepared.checks.size() + 2u) ||
        !sink.append(StatusMiddle) || !sink.append(CheckBody)) {
      return false;
    }
    for (std::size_t index = 0u; index < prepared.checks.size(); ++index) {
      const VulkanMapCheck check = prepared.checks[index];
      if (!sink.append(CheckLinePrefix) ||
          !backend_source_recipe::append_decimal(sink, index) ||
          !sink.append(CheckLineOffset) ||
          !backend_source_recipe::append_decimal(sink, check.offset) ||
          !sink.append(CheckLineStride) ||
          !backend_source_recipe::append_decimal(sink, check.stride) ||
          !sink.append(CheckLineLimit) ||
          !backend_source_recipe::append_decimal(sink, check.limit) ||
          !sink.append(CheckLineSuffix)) {
        return false;
      }
    }
    return sink.append(CheckTail);
  }
};

[[nodiscard]] inline rund::kernel::LoweringArtifact
VulkanMapCheckArtifact(const VulkanMapTemplateResources &prepared) {
  const auto [hi, lo] = VulkanMapCheckHash(prepared);
  rund::kernel::LoweringArtifact artifact{};
  artifact.key.api = rund::kernel::ComputeApi::Vulkan;
  artifact.key.scalar = prepared.plan.scalar;
  artifact.key.domain = prepared.plan.domain;
  artifact.key.fixed_format = prepared.plan.fixed_format;
  artifact.key.op_hash_hi = hi;
  artifact.key.op_hash_lo = lo;
  artifact.key.canonical_ir_hash_hi = hi;
  artifact.key.canonical_ir_hash_lo = lo;
  artifact.kind = rund::kernel::LoweringArtifactKind::VulkanSource;
  if (prepared.checks.empty() ||
      prepared.checks.size() > rund::kernel::kMaxComputeBindingCount) {
    artifact.reason = "compute_pipeline_capacity";
    return artifact;
  }
  std::uint64_t offset_digits = 0u;
  std::uint64_t stride_digits = 0u;
  std::uint64_t limit_digits = 0u;
  for (const VulkanMapCheck check : prepared.checks) {
    if (!rund::kernel::checked::add(offset_digits,
                                    VulkanDecimalDigitCount(check.offset),
                                    offset_digits) ||
        !rund::kernel::checked::add(stride_digits,
                                    VulkanDecimalDigitCount(check.stride),
                                    stride_digits) ||
        !rund::kernel::checked::add(
            limit_digits, VulkanDecimalDigitCount(check.limit), limit_digits)) {
      artifact.reason = "compute_pipeline_capacity";
      return artifact;
    }
  }
  std::uint64_t source_upper = 0u;
  if (!VulkanMapCheckSourceUpperBytes(prepared.checks.size(), offset_digits,
                                      stride_digits, limit_digits,
                                      source_upper) ||
      source_upper > std::numeric_limits<std::size_t>::max()) {
    artifact.reason = "compute_pipeline_capacity";
    return artifact;
  }
  const VulkanMapCheckSourceRecipe recipe{prepared};
  std::uint64_t exact_source_bytes = 0u;
  if (!backend_source_recipe::bytes(recipe, exact_source_bytes) ||
      exact_source_bytes != source_upper) {
    artifact.reason = "compute_artifact_mismatch";
    return artifact;
  }
  artifact.source_text =
      backend_source_recipe::materialize(recipe, exact_source_bytes);
  if (artifact.source_text.empty()) {
    artifact.reason = "compute_pipeline_capacity";
    return artifact;
  }
  artifact.source_text_upper_bytes = source_upper;
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
  artifact.source_text_upper_bytes = VulkanMapControlSourceText().size();
  artifact.ok = !artifact.source_text.empty();
  artifact.reason = artifact.ok ? "ok" : "compute_pipeline_capacity";
  return artifact;
}

[[nodiscard]] inline rund::kernel::LoweringArtifact
VulkanControlledMapArtifact(rund::kernel::LoweringArtifact artifact,
                            const rund::kernel::ComputePlan &plan) {
  std::uint64_t source_upper = 0u;
  if (!VulkanControlledMapSourceUpperBytes(
          plan,
          std::max<std::uint64_t>(artifact.source_text.size(),
                                  artifact.source_text_upper_bytes),
          source_upper) ||
      source_upper > std::numeric_limits<std::size_t>::max()) {
    artifact.ok = false;
    artifact.reason = "compute_pipeline_capacity";
    return artifact;
  }
  try {
    using namespace vulkan_controlled_map_source_detail;
    if (!backend_source_recipe::reserve_string(artifact.source_text,
                                               source_upper)) {
      artifact.ok = false;
      artifact.reason = "compute_pipeline_capacity";
      return artifact;
    }
    std::uint64_t binding = 0u;
    constexpr std::size_t ControlledEntryCapacity =
        DeclarationPrefix.size() + 20u + DeclarationSuffix.size() +
        Entry.size();
    std::array<char, ControlledEntryCapacity> controlled_storage{};
    backend_source_recipe::FixedBufferSink<ControlledEntryCapacity>
        controlled_sink{controlled_storage};
    if (!rund::kernel::checked::add(plan.input_buffer_count,
                                    plan.output_buffer_count, binding) ||
        !rund::kernel::checked::add(binding, 1u, binding) ||
        !controlled_sink.append(DeclarationPrefix) ||
        !backend_source_recipe::append_decimal(controlled_sink, binding) ||
        !controlled_sink.append(DeclarationSuffix) ||
        !controlled_sink.append(Entry)) {
      artifact.ok = false;
      artifact.reason = "compute_pipeline_capacity";
      return artifact;
    }
    const std::size_t frozen_capacity = artifact.source_text.capacity();
    const auto replace_unique = [&](const std::string_view needle,
                                    const std::string_view replacement) {
      const std::size_t at = artifact.source_text.find(needle);
      if (needle.empty() || at == std::string::npos ||
          artifact.source_text.find(needle, at + needle.size()) !=
              std::string::npos) {
        return false;
      }
      artifact.source_text.replace(at, needle.size(), replacement.data(),
                                   replacement.size());
      return artifact.source_text.capacity() == frozen_capacity;
    };
    if (!replace_unique(Entry, controlled_sink.text()) ||
        !replace_unique(Guard, ControlledGuard) ||
        !replace_unique(CanonicalVariant, ControlledVariant) ||
        artifact.source_text.size() > source_upper ||
        artifact.source_text.capacity() != frozen_capacity) {
      artifact.ok = false;
      artifact.reason = "compute_artifact_mismatch";
      return artifact;
    }
    artifact.key.variant = rund::kernel::LoweringArtifactVariant::Controlled;
    artifact.source_text_upper_bytes = source_upper;
    return artifact;
  } catch (...) {
    backend_exception::RethrowUnlessCapacityException();
    artifact.ok = false;
    artifact.reason = "compute_pipeline_capacity";
    return artifact;
  }
}

[[nodiscard]] inline bool PrepareVulkanMapControl(
    const rund::AccelDevice &pick, const BoundControl &bound,
    const std::vector<rund::kernel::ComputeDispatchWindow> &windows,
    VulkanMapEncodeResources &resources) {
  if (resources.prepared == nullptr ||
      (!bound.active() && resources.prepared->checks.empty())) {
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
  resources.control_pipeline = resources.prepared->control_pipeline;
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
  if (resources.prepared->checks.empty()) {
    return true;
  }
  const std::uint64_t capacity =
      windows.back().begin_sequence + windows.back().tile_count;
  std::vector<VulkanStorageBinding> check_bindings;
  check_bindings.reserve(resources.prepared->checks.size() + 3u);
  check_bindings.push_back(count_binding);
  check_bindings.push_back(predicate_binding);
  resources.check_bases.clear();
  resources.check_bases.reserve(resources.prepared->checks.size());
  for (const VulkanMapCheck &check : resources.prepared->checks) {
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
        !rund::kernel::checked::add((capacity - 1u) * ref->stride_bytes,
                                    ref->element_bytes)) {
      return false;
    }
    const std::uint64_t base =
        ref->offset_bytes - ref->offset_bytes % alignment;
    if (ref->offset_bytes - base != check.offset ||
        ref->stride_bytes != check.stride) {
      return false;
    }
    resources.check_bases.push_back(base);
    const std::uint64_t payload =
        (capacity - 1u) * ref->stride_bytes + ref->element_bytes;
    if (check.offset > std::numeric_limits<std::uint64_t>::max() - payload) {
      return false;
    }
    const std::uint64_t range = check.offset + payload;
    if (base > resident.device_buffer->bytes ||
        range > resident.device_buffer->bytes - base ||
        range > resources.adapter->storage_limit) {
      return false;
    }
    check_bindings.push_back(
        VulkanStorageBinding{resident.device_buffer, base, range});
  }
  check_bindings.push_back(VulkanStorageBinding{
      &resources.control_status.device, 0u, 2u * sizeof(std::uint32_t)});
  resources.check_pipeline = resources.prepared->check_pipeline;
  return resources.check_pipeline != nullptr &&
         AcquireVulkanCollectiveDescriptorSet(
             *resources.adapter, *resources.check_pipeline,
             static_cast<std::uint32_t>(check_bindings.size()),
             resources.check_descriptor) &&
         WriteVulkanStorageDescriptorSet(
             *resources.adapter, resources.check_descriptor,
             check_bindings.data(),
             static_cast<std::uint32_t>(check_bindings.size()));
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
      map.prepared->checks.empty() ? 0u : 1u};
  return params;
}

inline void EncodeVulkanMapCheck(VkCommandBuffer command,
                                 const VulkanMapEncodeResources &map) {
  if (map.prepared == nullptr || map.prepared->checks.empty()) {
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
