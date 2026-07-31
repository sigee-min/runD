#pragma once

#include "../../../clock.hpp"
#include "../../runtime/counter.hpp"
#include "../../shader/module.hpp"
#include "../pipeline.hpp"
#include "compute.hpp"
#include "layout.hpp"

#include <string_view>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] bool CreateCollectivePipeline(
    VulkanAdapter &adapter, const std::uint32_t descriptor_count,
    const std::uint32_t push_bytes, const std::uint64_t source_hash,
    const rund::kernel::ArtifactKey &key, const std::string_view source,
    const VulkanShader &shader, const VulkanSpecialization &specialization,
    VulkanCollectivePipeline &pipeline) {
  VulkanModule module{};
  pipeline.device = adapter.device;
  pipeline.descriptor_count = descriptor_count;
  pipeline.push_bytes = push_bytes;
  pipeline.key = key;
  pipeline.source_hash = source_hash;
  pipeline.source.assign(source.data(), source.size());
  pipeline.specialization = specialization;
  const std::uint64_t pipeline_begin = MonotonicNanoseconds();
  if (!CreateVulkanModule(adapter, shader, module).ok ||
      !CreateCollectiveDescriptorSetLayout(adapter, descriptor_count,
                                           pipeline.descriptor_set_layout) ||
      !CreateCollectivePipelineLayout(adapter, pipeline.descriptor_set_layout,
                                      push_bytes, pipeline.pipeline_layout) ||
      !CreateCollectiveComputePipeline(adapter, module.get(),
                                       pipeline.pipeline_layout, specialization,
                                       pipeline.pipeline)) {
    return false;
  }
  RecordVulkanPipelineCreateNs(adapter,
                               MonotonicNanoseconds() - pipeline_begin);
  return true;
}

} // namespace
#endif

} // namespace rund::node::accel::detail
