#include "model.hpp"

#include <chrono>
#include <optional>
#include <thread>

namespace node_compute_cache_contract {
namespace {

[[nodiscard]] bool eviction_releases_lock_before_program_destruction() {
  auto device = rund::compute::detail::open_cpu(1u);
  if (!device) {
    return false;
  }
  auto cache = std::make_shared<rund::compute::detail::ProgramCacheState>();
  cache->device = std::move(device).value();
  cache->capacity = 1u;
  const auto gate = std::make_shared<DestructionGate>();
  const rund::compute::graph::Fingerprint first_key{.hi = 0xd351u, .lo = 1u};
  const rund::compute::graph::Fingerprint replacement_key{.hi = 0xd351u,
                                                          .lo = 2u};
  {
    const auto first =
        rund::compute::detail::cached_program(cache, first_key, [&] {
          return CachedResult::success(blocking_program_owner(gate));
        });
    if (!first) {
      return false;
    }
  }

  std::optional<CachedResult> replacement;
  std::thread evictor{[&] {
    replacement.emplace(
        rund::compute::detail::cached_program(cache, replacement_key, [] {
          return CachedResult::success(opaque_program_owner());
        }));
  }};
  bool destruction_entered = false;
  {
    std::unique_lock lock{gate->mutex};
    destruction_entered = gate->changed.wait_for(lock, std::chrono::seconds{5},
                                                 [&] { return gate->entered; });
  }
  const bool cache_lock_available = cache->mutex.try_lock();
  if (cache_lock_available) {
    cache->mutex.unlock();
  }
  {
    std::lock_guard lock{gate->mutex};
    gate->released = true;
  }
  gate->changed.notify_one();
  evictor.join();
  return destruction_entered && cache_lock_available && replacement &&
         *replacement;
}

[[nodiscard]] bool clear_releases_lock_before_program_destruction() {
  auto device = rund::compute::detail::open_cpu(1u);
  if (!device) {
    return false;
  }
  auto cache = std::make_shared<rund::compute::detail::ProgramCacheState>();
  cache->device = std::move(device).value();
  cache->capacity = 1u;
  const auto gate = std::make_shared<DestructionGate>();
  const rund::compute::graph::Fingerprint key{.hi = 0xc1eau, .lo = 1u};
  {
    const auto first = rund::compute::detail::cached_program(cache, key, [&] {
      return CachedResult::success(blocking_program_owner(gate));
    });
    if (!first) {
      return false;
    }
  }

  bool clear_returned = false;
  std::thread clearer{[&] {
    cache->clear_ready();
    clear_returned = true;
  }};
  bool destruction_entered = false;
  {
    std::unique_lock lock{gate->mutex};
    destruction_entered = gate->changed.wait_for(lock, std::chrono::seconds{5},
                                                 [&] { return gate->entered; });
  }
  const bool cache_lock_available = cache->mutex.try_lock();
  if (cache_lock_available) {
    cache->mutex.unlock();
  }
  {
    std::lock_guard lock{gate->mutex};
    gate->released = true;
  }
  gate->changed.notify_one();
  clearer.join();
  bool empty = false;
  {
    std::lock_guard lock{cache->mutex};
    empty = cache->entries.empty();
  }
  return destruction_entered && cache_lock_available && clear_returned && empty;
}

} // namespace

int RunLifetime() {
  if (!eviction_releases_lock_before_program_destruction()) {
    return 12;
  }
  if (!clear_releases_lock_before_program_destruction()) {
    return 13;
  }
  return 0;
}

} // namespace node_compute_cache_contract
