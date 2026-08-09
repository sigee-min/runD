#pragma once

#include "build.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline bool StoreMetalScanPipelineSet(
    MetalAdapter &adapter, const std::string &block_key,
    const std::string &prefix_key, const std::string &offset_key,
    const bool publish_block, const bool publish_prefix,
    const bool publish_offset, std::shared_ptr<void> &block,
    std::shared_ptr<void> &prefix, std::shared_ptr<void> &offset,
    const std::uint64_t create_ns) {
  const auto publish = [&adapter](const std::string &key,
                                  std::shared_ptr<void> &pipeline,
                                  const std::uint64_t elapsed) {
    MetalNamedPipelinePublishResult result =
        PublishMetalNamedPipeline(adapter, key, pipeline, elapsed);
    if (result.status != MetalNamedPipelinePublishStatus::Inserted) {
      RecordMetalUncachedPipelineCompile(adapter, elapsed);
    }
    pipeline = std::move(result.pipeline);
    return result.status != MetalNamedPipelinePublishStatus::Failed;
  };
  const bool block_ok = !publish_block || publish(block_key, block, create_ns);
  const bool prefix_ok =
      !publish_prefix ||
      publish(prefix_key, prefix, publish_block ? 0u : create_ns);
  const bool offset_ok =
      !publish_offset ||
      publish(offset_key, offset,
              publish_block || publish_prefix ? 0u : create_ns);
  return block_ok && prefix_ok && offset_ok;
}
#endif

} // namespace rund::node::accel::detail
