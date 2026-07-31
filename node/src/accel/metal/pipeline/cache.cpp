#include "cache.hpp"
#include "artifact/index.hpp"
#include "guard.hpp"
#include "../../source/hash.hpp"
#include <rund/counter.hpp>
#include <mutex>
#include <new>
#include <utility>

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] bool SameSource(const MetalSourceLibrary& cached,
                              const std::string_view source,
                              const std::uint64_t hash) noexcept {
  return cached.source_hash == hash && cached.source.size() == source.size() &&
         std::string_view{cached.source} == source &&
         cached.library != nullptr;
}

[[nodiscard]] std::size_t FindMetalSourceLibrary(
    const std::vector<MetalSourceLibrary>& libraries,
    const std::string_view source,
    const std::uint64_t hash) noexcept {
  for (std::size_t index = 0u; index < libraries.size(); ++index) {
    if (SameSource(libraries[index], source, hash)) {
      return index;
    }
  }
  return libraries.size();
}

[[nodiscard]] std::shared_ptr<void> PromoteMetalSourceLibrary(
    std::vector<MetalSourceLibrary>& libraries,
    const std::size_t index) {
  MetalSourceLibrary hit = std::move(libraries[index]);
  libraries.erase(libraries.begin() + static_cast<std::ptrdiff_t>(index));
  libraries.push_back(std::move(hit));
  return libraries.back().library;
}

}  // namespace

std::shared_ptr<void> LookupMetalNamedPipeline(
    MetalAdapter& adapter,
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
          adapter.stats.pipeline_cache_hit_count, 1u);
      return adapter.named_pipelines[index->second].pipeline;
    }
  }
  return {};
}

void StoreMetalNamedPipeline(MetalAdapter& adapter,
                             std::string key,
                             std::shared_ptr<void> pipeline,
                             const std::uint64_t create_ns) {
  key = MetalPipelineCacheKey(key);
  std::lock_guard<std::mutex> lock{adapter.mutex};
  if (pipeline == nullptr) {
    SetMetalLastError(adapter, "accel_metal_pipeline_unavailable");
    return;
  }
  const std::uint64_t hash = SourceHash(key);
  const auto [begin, end] = adapter.pipeline_index->named.equal_range(hash);
  for (auto index = begin; index != end; ++index) {
    if (index->second < adapter.named_pipelines.size() &&
        adapter.named_pipelines[index->second].name == key) {
      return;
    }
  }
  const std::size_t index = adapter.named_pipelines.size();
  try {
    adapter.named_pipelines.push_back(
        MetalNamedPipeline{std::move(key), std::move(pipeline)});
    try {
      adapter.pipeline_index->named.emplace(hash, index);
    } catch (const std::bad_alloc &) {
      adapter.named_pipelines.pop_back();
      throw;
    }
  } catch (const std::bad_alloc &) {
    SetMetalLastError(adapter, "compute_pipeline_capacity");
    return;
  }
  ::rund::detail::counter::Accumulate(adapter.stats.pipeline_compile_count, 1u);
  ::rund::detail::counter::Accumulate(adapter.stats.pipeline_create_ns,
                                      create_ns);
}

std::shared_ptr<void> LookupMetalSourceLibrary(
    MetalAdapter& adapter,
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

std::shared_ptr<void> PublishMetalSourceLibrary(
    MetalAdapter& adapter,
    std::string source,
    std::shared_ptr<void> library,
    const std::uint64_t compile_ns) {
  if (library == nullptr) {
    return {};
  }
  std::lock_guard<std::mutex> lock{adapter.mutex};
  ::rund::detail::counter::Accumulate(adapter.stats.library_compile_count, 1u);
  ::rund::detail::counter::Accumulate(adapter.stats.shader_compile_ns,
                                      compile_ns);
  const std::uint64_t hash = SourceHash(source);
  const std::size_t index =
      FindMetalSourceLibrary(adapter.source_libraries, source, hash);
  if (index != adapter.source_libraries.size()) {
    ::rund::detail::counter::Accumulate(adapter.stats.library_cache_hit_count,
                                        1u);
    return PromoteMetalSourceLibrary(adapter.source_libraries, index);
  }
  if (adapter.source_libraries.size() == kMetalSourceLibraryCapacity) {
    adapter.source_libraries.erase(adapter.source_libraries.begin());
  }
  adapter.source_libraries.push_back(
      MetalSourceLibrary{hash, std::move(source), std::move(library)});
  return adapter.source_libraries.back().library;
}

}  // namespace rund::node::accel::detail
