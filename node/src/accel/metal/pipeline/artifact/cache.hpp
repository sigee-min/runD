#pragma once

#include "../../state.hpp"
#include "../../../source/hash.hpp"
#include <rund/counter.hpp>
#include "index.hpp"
#include <memory>
#include <mutex>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

[[nodiscard]] std::shared_ptr<void> FindMetalMapArtifactPipeline(
    MetalAdapter& adapter,
    const rund::kernel::LoweringArtifact& artifact) {
  std::lock_guard<std::mutex> lock{adapter.mutex};
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
  return {};
}

#endif

}  // namespace rund::node::accel::detail
