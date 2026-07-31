#pragma once

#include "../../state.hpp"
#include "../../../source/hash.hpp"
#include <rund/counter.hpp>
#include "index.hpp"
#include <memory>
#include <mutex>
#include <new>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

[[nodiscard]] std::shared_ptr<void> StoreMetalMapArtifactPipeline(
    MetalAdapter& adapter,
    const rund::kernel::LoweringArtifact& artifact,
    const std::shared_ptr<void>& pipeline) {
  std::lock_guard<std::mutex> lock{adapter.mutex};
  if (pipeline == nullptr) {
    SetMetalLastError(adapter, "accel_metal_pipeline_unavailable");
    return {};
  }
  const std::uint64_t source_hash =
      SourceHash(artifact.source_text);
  const auto [begin, end] =
      adapter.pipeline_index->entries.equal_range(source_hash);
  for (auto entry = begin; entry != end; ++entry) {
    if (entry->second >= adapter.pipelines.size()) {
      continue;
    }
    const MetalPipeline &cached = adapter.pipelines[entry->second];
    if (cached.source_hash == source_hash && cached.key == artifact.key &&
        cached.source == artifact.source_text &&
        cached.pipeline != nullptr) {
      ::rund::detail::counter::Accumulate(
          adapter.stats.pipeline_cache_hit_count, 1u);
      return cached.pipeline;
    }
  }
  const std::size_t index = adapter.pipelines.size();
  try {
    adapter.pipelines.push_back(MetalPipeline{
        .key = artifact.key,
        .source_hash = source_hash,
        .source = artifact.source_text,
        .pipeline = pipeline,
    });
    try {
      adapter.pipeline_index->entries.emplace(source_hash, index);
    } catch (const std::bad_alloc &) {
      adapter.pipelines.pop_back();
      throw;
    }
  } catch (const std::bad_alloc &) {
    SetMetalLastError(adapter, "compute_pipeline_capacity");
    return {};
  }
  ::rund::detail::counter::Accumulate(adapter.stats.pipeline_compile_count, 1u);
  return pipeline;
}

#endif

}  // namespace rund::node::accel::detail
