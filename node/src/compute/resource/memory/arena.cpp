#include "model.hpp"

#include "arena/local.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace rund::compute::detail::resource_detail::memory_detail {

bool pack(const std::span<const graph::Resource> resources,
          const std::span<const Lifetime> lifetimes,
          const std::span<const std::uint32_t> ids,
          const std::uint64_t limit_bytes, const bool destructive,
          Layout &result) {
  if (limit_bytes < Alignment || limit_bytes % Alignment != 0u ||
      resources.size() != lifetimes.size()) {
    return false;
  }
  result = {};
  result.owners.resize(resources.size());
  result.offsets.resize(resources.size());
  const std::uint64_t chunk_bytes = std::min(Chunk, limit_bytes);
  std::size_t ordinary_count = 0u;
  for (const std::uint32_t id : ids) {
    if (id == 0u || id > resources.size() || resources[id - 1u].bytes == 0u ||
        resources[id - 1u].bytes > limit_bytes) {
      return false;
    }
    ordinary_count += resources[id - 1u].bytes <= chunk_bytes ? 1u : 0u;
  }
  std::vector<std::uint32_t> grouped(ids.size());
  std::size_t ordinary_cursor = 0u;
  std::size_t large_cursor = ordinary_count;
  for (const std::uint32_t id : ids) {
    grouped[resources[id - 1u].bytes <= chunk_bytes ? ordinary_cursor++
                                                    : large_cursor++] = id;
  }
  const std::span<const std::uint32_t> groups{grouped};
  const auto ordinary = groups.first(ordinary_count);
  const auto large = groups.subspan(ordinary_count);

  Placement placement;
  Arena arena;
  if (!ordinary.empty() &&
      (!place_ordinary(resources, lifetimes, ordinary, chunk_bytes, destructive,
                       result.offsets, placement) ||
       !measure_ordinary(placement, resources, ordinary, result.offsets,
                         chunk_bytes, arena))) {
    return false;
  }
  result.bytes = arena.bytes;
  result.count = arena.count;
  for (const std::uint32_t id : ordinary) {
    const std::uint64_t offset = result.offsets[id - 1u];
    const std::size_t chunk = static_cast<std::size_t>(offset / chunk_bytes);
    if (chunk >= arena.owners.size() || arena.owners[chunk] == 0u) {
      return false;
    }
    result.owners[id - 1u] = arena.owners[chunk];
    result.offsets[id - 1u] = offset % chunk_bytes;
  }
  return place_large(resources, lifetimes, large, destructive, result);
}

} // namespace rund::compute::detail::resource_detail::memory_detail
