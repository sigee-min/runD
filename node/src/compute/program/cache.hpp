#pragma once

#include <rund/compute/cache.hpp>
#include <rund/compute/graph/info.hpp>

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <variant>

namespace rund::compute::detail {

struct DeviceState;
struct ProgramState;

struct FingerprintLess {
  [[nodiscard]] bool
  operator()(const graph::Fingerprint &left,
             const graph::Fingerprint &right) const noexcept {
    if (left.hi != right.hi) {
      return left.hi < right.hi;
    }
    return left.lo < right.lo;
  }
};

struct ProgramCachePending final {};
using ProgramCacheOutcome =
    std::variant<ProgramCachePending, std::shared_ptr<ProgramState>, Status>;

struct ProgramCacheEntry;
using ProgramCacheEntries =
    std::map<graph::Fingerprint, std::shared_ptr<ProgramCacheEntry>,
             FingerprintLess>;

struct ProgramCacheEntry final {
  std::condition_variable ready;
  ProgramCacheOutcome outcome{ProgramCachePending{}};
  ProgramCacheEntries::iterator slot{};
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

using ProgramBuilder = std::function<Result<std::shared_ptr<ProgramState>>()>;

[[nodiscard]] Result<std::shared_ptr<ProgramState>>
cached_program(const std::shared_ptr<ProgramCacheState> &cache,
               graph::Fingerprint fingerprint, ProgramBuilder builder);

} // namespace rund::compute::detail
