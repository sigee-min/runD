#pragma once

#include <rund/storage.hpp>

#include <cstdint>
#include <string>

namespace rund::replay {

enum class StorageMode : std::uint8_t {
  Memory = 0u,
  Spill = 1u,
};

struct Storage final {
  StorageMode mode = StorageMode::Memory;
  std::string directory{};
  std::uint64_t cached_bytes = 4u * 1024u * 1024u;
  std::uint64_t segment_bytes = 64u * 1024u * 1024u;
  std::uint64_t max_bytes = 1024u * 1024u * 1024u;
  std::uint64_t max_allocated_bytes = 2ull * 1024ull * 1024ull * 1024ull;
  std::uint64_t minimum_free_bytes = 0u;
  ::rund::storage::Budget budget{};
};

struct StorageReport final {
  StorageMode mode = StorageMode::Memory;
  std::uint64_t logical_bytes = 0u;
  std::uint64_t encoded_bytes = 0u;
  std::uint64_t retained_bytes = 0u;
  std::uint64_t copied_bytes = 0u;
  std::uint64_t cached_bytes = 0u;
  std::uint64_t physical_bytes = 0u;
  std::uint64_t allocated_bytes = 0u;
  std::uint64_t reserved_bytes = 0u;
  std::uint64_t growths = 0u;
  std::uint64_t chunk_count = 0u;
  std::uint64_t segment_count = 0u;
  std::uint64_t cache_hits = 0u;
  std::uint64_t cache_misses = 0u;
  std::uint64_t cache_evictions = 0u;
};

struct Diagnostic final {
  std::uint64_t window_bytes = 0u;
  std::uint64_t window_records = 0u;
};

struct DiagnosticReport final {
  std::uint64_t retained_bytes = 0u;
  std::uint64_t retained_records = 0u;
  std::uint64_t evicted_records = 0u;
  std::uint64_t dropped_records = 0u;
};

} // namespace rund::replay
