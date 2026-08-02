#include "pipeline.hpp"
#include "index.hpp"

#include "../../clock.hpp"
#include "../../source/hash.hpp"
#include "../runtime/counter.hpp"
#include "../shader/module.hpp"
#include "pipeline/compute.hpp"
#include "pipeline/descriptor.hpp"
#include <rund/counter.hpp>

#include <new>
#include <utility>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
VulkanCachedPipeline::~VulkanCachedPipeline() { reset(); }

VulkanCachedPipeline::VulkanCachedPipeline(
    VulkanCachedPipeline &&other) noexcept {
  *this = std::move(other);
}

VulkanCachedPipeline &
VulkanCachedPipeline::operator=(VulkanCachedPipeline &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  reset();
  device = std::exchange(other.device, VK_NULL_HANDLE);
  key = other.key;
  source_hash = other.source_hash;
  source = std::move(other.source);
  input_buffer_count = other.input_buffer_count;
  output_buffer_count = other.output_buffer_count;
  shader_hash = other.shader_hash;
  descriptor_set_layout =
      std::exchange(other.descriptor_set_layout, VK_NULL_HANDLE);
  descriptor_pool = std::exchange(other.descriptor_pool, VK_NULL_HANDLE);
  descriptor_set = std::exchange(other.descriptor_set, VK_NULL_HANDLE);
  pipeline_layout = std::exchange(other.pipeline_layout, VK_NULL_HANDLE);
  pipeline = std::exchange(other.pipeline, VK_NULL_HANDLE);
  return *this;
}

void VulkanCachedPipeline::reset() noexcept {
  if (device != VK_NULL_HANDLE) {
    if (pipeline != VK_NULL_HANDLE) {
      vkDestroyPipeline(device, pipeline, nullptr);
    }
    if (pipeline_layout != VK_NULL_HANDLE) {
      vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    }
    if (descriptor_pool != VK_NULL_HANDLE) {
      vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
    }
    if (descriptor_set_layout != VK_NULL_HANDLE) {
      vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
    }
  }
  device = VK_NULL_HANDLE;
  key = {};
  source_hash = 0u;
  source.clear();
  input_buffer_count = 0u;
  output_buffer_count = 0u;
  shader_hash = 0u;
  descriptor_set_layout = VK_NULL_HANDLE;
  descriptor_pool = VK_NULL_HANDLE;
  descriptor_set = VK_NULL_HANDLE;
  pipeline_layout = VK_NULL_HANDLE;
  pipeline = VK_NULL_HANDLE;
}

bool VulkanCachedPipelineMatches(
    const VulkanCachedPipeline &pipeline,
    const rund::kernel::LoweringArtifact &artifact,
    const std::uint64_t source_hash, const rund::kernel::u64 input_buffer_count,
    const rund::kernel::u64 output_buffer_count) noexcept {
  return pipeline.source_hash == source_hash && pipeline.key == artifact.key &&
         pipeline.source == artifact.source_text &&
         pipeline.input_buffer_count == input_buffer_count &&
         pipeline.output_buffer_count == output_buffer_count &&
         pipeline.pipeline != VK_NULL_HANDLE &&
         pipeline.pipeline_layout != VK_NULL_HANDLE &&
         pipeline.descriptor_set_layout != VK_NULL_HANDLE;
}

VulkanCachedPipeline *
AcquireVulkanCachedPipeline(VulkanAdapter &adapter,
                            const rund::kernel::ComputePlan &plan,
                            rund::kernel::LoweringArtifact artifact) {
  const std::uint64_t source_hash = SourceHash(artifact.source_text);
  const auto [begin, end] =
      adapter.pipeline_index->entries.equal_range(source_hash);
  for (auto entry = begin; entry != end; ++entry) {
    VulkanCachedPipeline *const cached = entry->second;
    if (cached != nullptr &&
        VulkanCachedPipelineMatches(*cached, artifact, source_hash,
                                    plan.input_buffer_count,
                                    plan.output_buffer_count)) {
      ::rund::detail::counter::Accumulate(adapter.pipeline_cache_hit_count, 1u);
      return cached;
    }
  }
  VulkanCachedPipeline pipeline{};
  try {
    VulkanShader shader{};
    if (!CompileVulkanShader(adapter, plan, artifact, shader)) {
      return nullptr;
    }
    if (!CreateVulkanCachedPipeline(adapter, plan, std::move(artifact), shader,
                                    pipeline)) {
      return nullptr;
    }
    pipeline.source_hash = source_hash;
    adapter.pipelines.push_back(std::move(pipeline));
    pipeline = {};
    VulkanCachedPipeline *const stored = &adapter.pipelines.back();
    try {
      adapter.pipeline_index->entries.emplace(source_hash, stored);
    } catch (...) {
      adapter.pipelines.pop_back();
      throw;
    }
    ::rund::detail::counter::Accumulate(adapter.pipeline_compile_count, 1u);
    return stored;
  } catch (const std::bad_alloc &) {
    return nullptr;
  }
}

bool CreateVulkanCachedPipeline(VulkanAdapter &adapter,
                                const rund::kernel::ComputePlan &plan,
                                rund::kernel::LoweringArtifact artifact,
                                const VulkanShader &shader,
                                VulkanCachedPipeline &pipeline) {
  VulkanModule module{};
  pipeline.device = adapter.device;
  pipeline.key = artifact.key;
  pipeline.source = std::move(artifact.source_text);
  pipeline.input_buffer_count = plan.input_buffer_count;
  pipeline.output_buffer_count = plan.output_buffer_count;
  pipeline.shader_hash = shader.hash;
  const std::uint64_t pipeline_begin = MonotonicNanoseconds();
  std::uint32_t descriptor_count = 0u;
  if (!CreateVulkanModule(adapter, shader, module).ok ||
      !VulkanDescriptorCountForPlan(adapter, plan, descriptor_count) ||
      !CreateVulkanDescriptorLayout(adapter, descriptor_count, pipeline) ||
      !CreateVulkanPipelineLayout(adapter, pipeline) ||
      !CreateVulkanComputePipeline(adapter, module.get(), pipeline)) {
    return false;
  }
  RecordVulkanPipelineCreateNs(adapter,
                               MonotonicNanoseconds() - pipeline_begin);
  return true;
}
#endif

} // namespace rund::node::accel::detail
