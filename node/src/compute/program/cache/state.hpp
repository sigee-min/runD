#pragma once

#include <rund/compute/graph/info.hpp>
#include <rund/compute/status.hpp>

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <variant>

namespace rund::compute::detail {

struct DeviceState;
struct ProgramState;

struct FingerprintHash {
  [[nodiscard]] std::size_t
  operator()(const graph::Fingerprint &key) const noexcept {
    const auto mix = [](std::uint64_t value) noexcept {
      value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
      value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
      return value ^ (value >> 31u);
    };
    return static_cast<std::size_t>(mix(key.hi) ^
                                    mix(key.lo + 0x9e3779b97f4a7c15ull));
  }
};

struct ProgramCachePending final {};
using ProgramCacheOutcome =
    std::variant<ProgramCachePending, std::shared_ptr<ProgramState>, Status>;

struct ProgramCacheEntry;
using ProgramCacheEntries =
    std::unordered_map<graph::Fingerprint, std::shared_ptr<ProgramCacheEntry>,
                       FingerprintHash>;

struct ProgramCacheEntry final {
  std::condition_variable ready;
  ProgramCacheOutcome outcome{ProgramCachePending{}};
  graph::Fingerprint key{};
  std::shared_ptr<ProgramCacheEntry> retired_next{};
  ProgramCacheEntry *older{};
  ProgramCacheEntry *newer{};
};

struct ProgramCacheState final {
  std::shared_ptr<DeviceState> device;
  std::size_t capacity{};
  mutable std::mutex mutex;
  ProgramCacheEntries entries;
  ProgramCacheEntry *oldest{};
  ProgramCacheEntry *newest{};
  std::size_t ready_count{};
  std::uint64_t hits{};
  std::uint64_t misses{};
  std::uint64_t waits{};
  std::uint64_t evictions{};

  void clear_ready() noexcept;
};

} // namespace rund::compute::detail
