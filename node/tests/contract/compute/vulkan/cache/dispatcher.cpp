#include "support.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
#include "../../../../../src/accel/source/hash.hpp"
#include "../../../../../src/accel/vulkan/adapter/state.hpp"
#include "../../../../../src/accel/vulkan/cached/index.hpp"
#include "../../../../../src/accel/vulkan/cached/pipeline.hpp"
#include "../../../../../src/accel/vulkan/collective/pipeline.hpp"
#include "../../../../../src/accel/vulkan/descriptor.hpp"
#include "../../../../../src/accel/vulkan/kernel/pipeline/source.hpp"
#include "../../../../../src/accel/vulkan/shader/cache.hpp"
#include "../../../../../src/accel/vulkan/shader/module.hpp"

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/binding/model.hpp>
#include <rund/counter.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <type_traits>
#endif

int RunComputeVulkanCacheContract() {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  using rund::node::accel::detail::VulkanAdapter;
  using rund::node::accel::detail::VulkanCachedPipeline;
  using rund::node::accel::detail::VulkanCollectivePipeline;
  using rund::node::accel::detail::VulkanModule;
  using rund::node::accel::detail::VulkanShader;

  static_assert(!std::is_copy_constructible_v<VulkanModule>);
  static_assert(!std::is_copy_assignable_v<VulkanModule>);
  static_assert(std::is_nothrow_move_constructible_v<VulkanModule>);
  static_assert(std::is_nothrow_move_assignable_v<VulkanModule>);
  static_assert(!std::is_copy_constructible_v<VulkanCachedPipeline>);
  static_assert(!std::is_copy_assignable_v<VulkanCachedPipeline>);
  static_assert(std::is_nothrow_move_constructible_v<VulkanCachedPipeline>);
  static_assert(std::is_nothrow_move_assignable_v<VulkanCachedPipeline>);
  static_assert(!std::is_copy_constructible_v<VulkanCollectivePipeline>);
  static_assert(!std::is_copy_assignable_v<VulkanCollectivePipeline>);
  static_assert(std::is_nothrow_move_constructible_v<VulkanCollectivePipeline>);
  static_assert(std::is_nothrow_move_assignable_v<VulkanCollectivePipeline>);

  constexpr std::uint64_t telemetry_hash_hi = 0x706970652e74656cull;
  constexpr std::uint64_t telemetry_hash_lo = 0x656d657472792e31ull;
  constexpr std::uint64_t profile_hash_hi = 0x706970652e70726full;
  constexpr std::uint64_t profile_hash_lo = 0x66696c652e763100ull;
  const std::string_view telemetry_source =
      rund::node::accel::detail::VulkanTelemetrySourceText();
  const std::string profile_source =
      rund::node::accel::detail::VulkanProfileSource();
  const auto telemetry_plan = rund::node::accel::detail::VulkanTelemetryPlan();
  const auto profile_plan = rund::node::accel::detail::VulkanProfilePlan();
  if (telemetry_source.empty() || profile_source.empty() ||
      telemetry_source == profile_source ||
      rund::node::accel::detail::VulkanProfileSourceBytes() !=
          profile_source.size() ||
      telemetry_plan.op_hash_hi != telemetry_hash_hi ||
      telemetry_plan.op_hash_lo != telemetry_hash_lo ||
      profile_plan.op_hash_hi != profile_hash_hi ||
      profile_plan.op_hash_lo != profile_hash_lo ||
      telemetry_plan.api != rund::kernel::ComputeApi::Vulkan ||
      profile_plan.api != rund::kernel::ComputeApi::Vulkan ||
      !telemetry_plan.ok || !profile_plan.ok) {
    return 40;
  }

  if (const int descriptor = rund::node::vulkan_cache_contract::DescriptorRange();
      descriptor != 0) {
    return descriptor;
  }
  if (const int bias = rund::node::vulkan_cache_contract::MapBias(); bias != 0) {
    return bias;
  }

  VulkanAdapter adapter{};
  adapter.pipelines.push_back(VulkanCachedPipeline{});
  VulkanCachedPipeline *const first = &adapter.pipelines.front();
  for (std::size_t index = 0u; index < 128u; ++index) {
    adapter.pipelines.push_back(VulkanCachedPipeline{});
  }
  if (&adapter.pipelines.front() != first) {
    return 1;
  }

  adapter.collective_pipelines.emplace_back();
  adapter.collective_pipelines.emplace_back();
  VulkanCollectivePipeline &used = adapter.collective_pipelines.front();
  VulkanCollectivePipeline &idle = adapter.collective_pipelines.back();
  used.descriptor_epoch = adapter.pipeline_index->descriptor_epoch;
  idle.descriptor_epoch = adapter.pipeline_index->descriptor_epoch;
  used.next_descriptor_slot = 7u;
  idle.next_descriptor_slot = 11u;
  used.descriptor_sets.resize(3u, VK_NULL_HANDLE);
  rund::node::accel::detail::BeginVulkanCollectiveDescriptorEpoch(adapter);
  if (used.next_descriptor_slot != 7u || idle.next_descriptor_slot != 11u) {
    return 26;
  }
  rund::node::accel::detail::PrepareVulkanCollectiveDescriptorSlots(adapter,
                                                                     used);
  if (used.next_descriptor_slot != 0u || used.reusable_descriptor_count != 3u ||
      idle.next_descriptor_slot != 11u) {
    return 27;
  }
  used.next_descriptor_slot = 2u;
  rund::node::accel::detail::PrepareVulkanCollectiveDescriptorSlots(adapter,
                                                                     used);
  if (used.next_descriptor_slot != 2u) {
    return 28;
  }
  adapter.pipeline_index->descriptor_epoch =
      std::numeric_limits<std::uint64_t>::max();
  used.descriptor_epoch = adapter.pipeline_index->descriptor_epoch;
  idle.descriptor_epoch = adapter.pipeline_index->descriptor_epoch;
  rund::node::accel::detail::BeginVulkanCollectiveDescriptorEpoch(adapter);
  if (adapter.pipeline_index->descriptor_epoch != 1u ||
      used.descriptor_epoch != 0u || idle.descriptor_epoch != 0u) {
    return 29;
  }

  const rund::kernel::ArtifactKey exact_key{
      .api = rund::kernel::ComputeApi::Vulkan,
      .op_hash_hi = 17u,
      .op_hash_lo = 29u,
      .canonical_ir_hash_hi = 41u,
      .canonical_ir_hash_lo = 53u,
  };
  const rund::kernel::LoweringArtifact exact_artifact{
      .key = exact_key,
      .source_text = "exact-source-a",
  };
  const rund::kernel::LoweringArtifact colliding_artifact{
      .key = exact_key,
      .source_text = "exact-source-b",
  };
  rund::kernel::ArtifactKey distinct_key = exact_key;
  distinct_key.op_hash_lo += 1u;
  const rund::kernel::LoweringArtifact distinct_artifact{
      .key = distinct_key,
      .source_text = exact_artifact.source_text,
  };
  const auto handle = [](const std::uintptr_t value) {
    return reinterpret_cast<void *>(value);
  };
  VulkanCachedPipeline exact_pipeline{};
  exact_pipeline.key = exact_key;
  exact_pipeline.source_hash =
      rund::node::accel::detail::SourceHash(exact_artifact.source_text);
  exact_pipeline.source = exact_artifact.source_text;
  exact_pipeline.input_buffer_count = 2u;
  exact_pipeline.output_buffer_count = 1u;
  exact_pipeline.descriptor_set_layout =
      reinterpret_cast<VkDescriptorSetLayout>(handle(1u));
  exact_pipeline.pipeline_layout =
      reinterpret_cast<VkPipelineLayout>(handle(2u));
  exact_pipeline.pipeline = reinterpret_cast<VkPipeline>(handle(3u));
  const std::uint64_t exact_hash =
      rund::node::accel::detail::SourceHash(exact_artifact.source_text);
  if (!rund::node::accel::detail::VulkanCachedPipelineMatches(
          exact_pipeline, exact_artifact, exact_hash, 2u, 1u) ||
      rund::node::accel::detail::VulkanCachedPipelineMatches(
          exact_pipeline, colliding_artifact, exact_hash, 2u, 1u) ||
      rund::node::accel::detail::VulkanCachedPipelineMatches(
          exact_pipeline, distinct_artifact, exact_hash, 2u, 1u) ||
      rund::node::accel::detail::VulkanCachedPipelineMatches(
          exact_pipeline, exact_artifact, exact_hash, 1u, 1u) ||
      rund::node::accel::detail::VulkanCachedPipelineMatches(
          exact_pipeline, exact_artifact, exact_hash, 2u, 2u)) {
    return 23;
  }
  VulkanCachedPipeline moved_pipeline{std::move(exact_pipeline)};
  if (exact_pipeline.pipeline != VK_NULL_HANDLE ||
      exact_pipeline.pipeline_layout != VK_NULL_HANDLE ||
      exact_pipeline.descriptor_set_layout != VK_NULL_HANDLE ||
      moved_pipeline.pipeline == VK_NULL_HANDLE ||
      moved_pipeline.pipeline_layout == VK_NULL_HANDLE ||
      moved_pipeline.descriptor_set_layout == VK_NULL_HANDLE) {
    return 24;
  }

  if (const int shader = rund::node::vulkan_cache_contract::VulkanShaderCache();
      shader != 0) {
    return shader;
  }
  if (const int source_identity =
          rund::node::vulkan_cache_contract::VulkanCollectiveSourceIdentity();
      source_identity != 0) {
    return source_identity;
  }
#endif
  return 0;
}
