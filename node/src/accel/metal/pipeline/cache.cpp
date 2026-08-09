#include "cache.hpp"
#include "../../source/hash.hpp"
#include "artifact/index.hpp"
#include "guard.hpp"
#include <algorithm>
#include <mutex>
#include <new>
#include <rund/counter.hpp>
#include <type_traits>
#include <utility>

namespace rund::node::accel::detail {
namespace {

static_assert(std::is_nothrow_move_constructible_v<MetalSourceLibrary>);
static_assert(std::is_nothrow_move_assignable_v<MetalSourceLibrary>);
static_assert(std::is_nothrow_move_constructible_v<MetalNamedPipeline>);
static_assert(std::is_nothrow_move_assignable_v<MetalNamedPipeline>);

[[nodiscard]] bool SameSource(const MetalSourceLibrary &cached,
                              const std::string_view source,
                              const std::uint64_t hash) noexcept {
  return cached.source_hash == hash && cached.source.size() == source.size() &&
         std::string_view{cached.source} == source && cached.library != nullptr;
}

[[nodiscard]] std::size_t
FindMetalSourceLibrary(const std::vector<MetalSourceLibrary> &libraries,
                       const std::string_view source,
                       const std::uint64_t hash) noexcept {
  for (std::size_t index = 0u; index < libraries.size(); ++index) {
    if (SameSource(libraries[index], source, hash)) {
      return index;
    }
  }
  return libraries.size();
}

[[nodiscard]] std::shared_ptr<void>
PromoteMetalSourceLibrary(std::vector<MetalSourceLibrary> &libraries,
                          const std::size_t index) noexcept {
  MetalSourceLibrary hit = std::move(libraries[index]);
  libraries.erase(libraries.begin() + static_cast<std::ptrdiff_t>(index));
  libraries.push_back(std::move(hit));
  return libraries.back().library;
}

} // namespace

std::shared_ptr<void> LookupMetalNamedPipeline(MetalAdapter &adapter,
                                               const std::string_view key) {
  const std::string scoped_key = MetalPipelineCacheKey(key);
  std::lock_guard<std::mutex> lock{adapter.mutex};
  const std::uint64_t hash = SourceHash(scoped_key);
  const auto [begin, end] = adapter.pipeline_index->named.equal_range(hash);
  for (auto index = begin; index != end; ++index) {
    if (index->second < adapter.named_pipelines.size() &&
        adapter.named_pipelines[index->second].name == scoped_key &&
        adapter.named_pipelines[index->second].pipeline != nullptr) {
      ::rund::detail::counter::Accumulate(
          adapter.stats.runtime.run.allocations.pipeline_cache_hit_count, 1u);
      return adapter.named_pipelines[index->second].pipeline;
    }
  }
  return {};
}

MetalNamedPipelinePublishResult
PublishMetalNamedPipeline(MetalAdapter &adapter, std::string key,
                          std::shared_ptr<void> pipeline,
                          const std::uint64_t create_ns) noexcept {
  try {
    key = MetalPipelineCacheKey(key);
  } catch (...) {
    SetMetalLastError(adapter, "compute_pipeline_capacity");
    return {};
  }
  try {
    std::lock_guard<std::mutex> lock{adapter.mutex};
    if (pipeline == nullptr) {
      adapter.last_error = "accel_metal_pipeline_unavailable";
      return {};
    }
    const std::uint64_t hash = SourceHash(key);
    const auto [begin, end] = adapter.pipeline_index->named.equal_range(hash);
    for (auto index = begin; index != end; ++index) {
      if (index->second < adapter.named_pipelines.size() &&
          adapter.named_pipelines[index->second].name == key &&
          adapter.named_pipelines[index->second].pipeline != nullptr) {
        return {MetalNamedPipelinePublishStatus::Existing,
                adapter.named_pipelines[index->second].pipeline};
      }
    }
    if (adapter.fault_named_pipeline_publish_once.exchange(
            false, std::memory_order_relaxed)) {
      adapter.last_error = "compute_pipeline_capacity";
      return {};
    }
    const std::size_t index = adapter.named_pipelines.size();
    try {
      adapter.named_pipelines.push_back(
          MetalNamedPipeline{std::move(key), std::move(pipeline)});
      try {
        adapter.pipeline_index->named.emplace(hash, index);
      } catch (...) {
        adapter.named_pipelines.pop_back();
        throw;
      }
    } catch (...) {
      adapter.last_error = "compute_pipeline_capacity";
      return {};
    }
    ::rund::detail::counter::Accumulate(
        adapter.stats.runtime.run.allocations.pipeline_compile_count, 1u);
    ::rund::detail::counter::Accumulate(
        adapter.stats.runtime.run.time.pipeline_create_ns, create_ns);
    return {MetalNamedPipelinePublishStatus::Inserted,
            adapter.named_pipelines.back().pipeline};
  } catch (...) {
    return {};
  }
}

void StoreMetalNamedPipeline(MetalAdapter &adapter, std::string key,
                             std::shared_ptr<void> pipeline,
                             const std::uint64_t create_ns) {
  (void)PublishMetalNamedPipeline(adapter, std::move(key), std::move(pipeline),
                                  create_ns);
}

void RecordMetalUncachedPipelineCompile(
    MetalAdapter &adapter, const std::uint64_t create_ns) noexcept {
  std::lock_guard<std::mutex> lock{adapter.mutex};
  ::rund::detail::counter::Accumulate(
      adapter.stats.runtime.run.allocations.pipeline_compile_count, 1u);
  ::rund::detail::counter::Accumulate(
      adapter.stats.runtime.run.time.pipeline_create_ns, create_ns);
}

std::shared_ptr<void> LookupMetalSourceLibrary(MetalAdapter &adapter,
                                               const std::string_view source) {
  std::lock_guard<std::mutex> lock{adapter.mutex};
  const std::uint64_t hash = SourceHash(source);
  const std::size_t index =
      FindMetalSourceLibrary(adapter.source_libraries, source, hash);
  if (index != adapter.source_libraries.size()) {
    ::rund::detail::counter::Accumulate(adapter.stats.library_cache_hit_count,
                                        1u);
    return PromoteMetalSourceLibrary(adapter.source_libraries, index);
  }
  return {};
}

MetalSourceLibraryPublishResult
PublishMetalSourceLibrary(MetalAdapter &adapter, std::string source,
                          std::shared_ptr<void> library,
                          const std::uint64_t compile_ns) noexcept {
  try {
    std::lock_guard<std::mutex> lock{adapter.mutex};
    if (library == nullptr) {
      adapter.last_error = "accel_metal_pipeline_unavailable";
      return {};
    }

    // The caller invokes publication only for a library it just constructed.
    // Account that construction once regardless of which cache transaction
    // disposition follows.
    ::rund::detail::counter::Accumulate(adapter.stats.library_compile_count,
                                        1u);
    ::rund::detail::counter::Accumulate(
        adapter.stats.runtime.run.time.shader_compile_ns, compile_ns);
    const std::uint64_t hash = SourceHash(source);
    const std::size_t index =
        FindMetalSourceLibrary(adapter.source_libraries, source, hash);
    if (index != adapter.source_libraries.size()) {
      ::rund::detail::counter::Accumulate(adapter.stats.library_cache_hit_count,
                                          1u);
      return {MetalSourceLibraryPublishStatus::Existing,
              PromoteMetalSourceLibrary(adapter.source_libraries, index)};
    }
    if (adapter.fault_source_library_publish_once.exchange(
            false, std::memory_order_relaxed)) {
      adapter.last_error = "compute_pipeline_capacity";
      return {};
    }

    MetalSourceLibrary candidate{hash, std::move(source), std::move(library)};
    if (adapter.source_libraries.size() == kMetalSourceLibraryCapacity) {
      // A full cache already owns all backing slots. Rotate the LRU entry to
      // the back and replace it with noexcept moves, so publication cannot
      // erase the old owner and then fail while growing the vector.
      std::rotate(adapter.source_libraries.begin(),
                  adapter.source_libraries.begin() + 1,
                  adapter.source_libraries.end());
      adapter.source_libraries.back() = std::move(candidate);
    } else {
      try {
        adapter.source_libraries.push_back(std::move(candidate));
      } catch (...) {
        adapter.last_error = "compute_pipeline_capacity";
        return {};
      }
    }
    return {MetalSourceLibraryPublishStatus::Inserted,
            adapter.source_libraries.back().library};
  } catch (...) {
    // Mutex acquisition is the only operation outside the inner allocation
    // transaction. Publication is an explicit fail-closed, no-throw API.
    return {};
  }
}

void RecordMetalUncachedLibraryCompile(
    MetalAdapter &adapter, const std::uint64_t compile_ns) noexcept {
  std::lock_guard<std::mutex> lock{adapter.mutex};
  ::rund::detail::counter::Accumulate(adapter.stats.library_compile_count, 1u);
  ::rund::detail::counter::Accumulate(
      adapter.stats.runtime.run.time.shader_compile_ns, compile_ns);
}

} // namespace rund::node::accel::detail
