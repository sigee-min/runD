#pragma once

#include <rund/counter.hpp>
#include "../../../source/hash.hpp"
#include "../../cached/index.hpp"
#include "../pipeline.hpp"
#include "create.hpp"

#include <new>
#include <string_view>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
VulkanCollectivePipeline *AcquireVulkanCollectivePipeline(
    VulkanAdapter &adapter, const std::uint32_t descriptor_count,
    const std::uint32_t push_bytes, const rund::kernel::ComputePlan &plan,
    const rund::kernel::LoweringArtifact &artifact,
    const VulkanSpecialization &specialization) {
  const std::string_view source = artifact.source_text;
  const std::uint64_t source_hash = SourceHash(source);
  const auto [begin, end] =
      adapter.pipeline_index->collectives.equal_range(source_hash);
  for (auto entry = begin; entry != end; ++entry) {
    VulkanCollectivePipeline *const cached = entry->second;
    if (cached != nullptr && cached->key == artifact.key &&
        cached->descriptor_count == descriptor_count &&
        cached->push_bytes == push_bytes &&
        cached->specialization == specialization &&
        cached->source_hash == source_hash &&
        cached->source.size() == source.size() &&
        std::string_view{cached->source} == source &&
        cached->pipeline != VK_NULL_HANDLE &&
        cached->pipeline_layout != VK_NULL_HANDLE &&
        cached->descriptor_set_layout != VK_NULL_HANDLE) {
      ::rund::detail::counter::Accumulate(adapter.pipeline_cache_hit_count, 1u);
      return cached;
    }
  }

  VulkanCollectivePipeline pipeline{};
  try {
    VulkanShader shader{};
    if (!CompileVulkanSourceWithTools(adapter, plan, artifact, shader)) {
      return nullptr;
    }
    if (!CreateCollectivePipeline(adapter, descriptor_count, push_bytes,
                                  source_hash, artifact.key, source, shader,
                                  specialization, pipeline)) {
      return nullptr;
    }
    adapter.collective_pipelines.push_back(std::move(pipeline));
    VulkanCollectivePipeline *const stored =
        &adapter.collective_pipelines.back();
    try {
      adapter.pipeline_index->collectives.emplace(source_hash, stored);
    } catch (...) {
      adapter.collective_pipelines.pop_back();
      throw;
    }
    ::rund::detail::counter::Accumulate(adapter.pipeline_compile_count, 1u);
    return stored;
  } catch (const std::bad_alloc &) {
    return nullptr;
  }
}

#endif

} // namespace rund::node::accel::detail
