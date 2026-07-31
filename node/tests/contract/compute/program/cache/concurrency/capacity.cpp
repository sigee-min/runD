#include "model.hpp"

#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <thread>

namespace node_compute_cache_contract {
namespace {

[[nodiscard]] bool in_flight_success_restores_ready_capacity() {
  constexpr std::size_t capacity = 3u;
  constexpr std::size_t width = 12u;
  auto device = rund::compute::detail::open_cpu(1u);
  if (!device) {
    return false;
  }
  auto cache = std::make_shared<rund::compute::detail::ProgramCacheState>();
  cache->device = std::move(device).value();
  cache->capacity = capacity;

  std::mutex gate_mutex;
  std::condition_variable all_entered;
  std::condition_variable release_builders;
  std::size_t entered = 0u;
  bool released = false;
  std::array<std::optional<CachedResult>, width> results;
  std::array<std::thread, width> workers;
  for (std::size_t index = 0u; index < width; ++index) {
    workers[index] = std::thread{[&, index] {
      const auto build = [&] {
        {
          std::unique_lock lock{gate_mutex};
          ++entered;
          all_entered.notify_one();
          release_builders.wait(lock, [&] { return released; });
        }
        return CachedResult::success(opaque_program_owner());
      };
      results[index].emplace(rund::compute::detail::cached_program(
          cache, {.hi = 0xc001u, .lo = static_cast<std::uint64_t>(index + 1u)},
          build));
    }};
  }
  {
    std::unique_lock lock{gate_mutex};
    all_entered.wait(lock, [&] { return entered == width; });
  }
  bool all_in_flight = false;
  {
    std::lock_guard lock{cache->mutex};
    all_in_flight = cache->entries.size() == width;
  }
  {
    std::lock_guard lock{gate_mutex};
    released = true;
  }
  release_builders.notify_all();
  for (auto &worker : workers) {
    worker.join();
  }
  for (const auto &result : results) {
    if (!result || !*result) {
      return false;
    }
  }
  const rund::compute::ProgramCache::Stats stats{
      .hits = cache->hits,
      .misses = cache->misses,
      .waits = cache->waits,
      .evictions = cache->evictions,
      .ready_entries = cache->entries.size(),
      .in_flight = 0u,
      .capacity = cache->capacity,
  };
  return all_in_flight && stats.hits == 0u && stats.misses == width &&
         stats.waits == 0u && stats.evictions == width - capacity &&
         stats.ready_entries == capacity;
}

} // namespace

int RunCapacity() {
  return in_flight_success_restores_ready_capacity() ? 0 : 11;
}

} // namespace node_compute_cache_contract
