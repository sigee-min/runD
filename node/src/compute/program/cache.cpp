#include "cache.hpp"

#include <rund/counter.hpp>
#include "state.hpp"

#include <rund/compute/device.hpp>
#include <rund/compute/flow/builder.hpp>

#include <new>
#include <utility>

namespace rund::compute {
namespace detail {
namespace {

[[nodiscard]] bool linked(const ProgramCacheState &cache,
                          const ProgramCacheEntry &entry) noexcept {
  return cache.oldest == &entry || entry.older != nullptr ||
         entry.newer != nullptr;
}

void unlink(ProgramCacheState &cache, ProgramCacheEntry &entry) noexcept {
  if (!linked(cache, entry)) {
    return;
  }
  if (entry.older != nullptr) {
    entry.older->newer = entry.newer;
  } else {
    cache.oldest = entry.newer;
  }
  if (entry.newer != nullptr) {
    entry.newer->older = entry.older;
  } else {
    cache.newest = entry.older;
  }
  entry.older = nullptr;
  entry.newer = nullptr;
  --cache.ready_count;
}

void append(ProgramCacheState &cache, ProgramCacheEntry &entry) noexcept {
  entry.older = cache.newest;
  entry.newer = nullptr;
  if (cache.newest != nullptr) {
    cache.newest->newer = &entry;
  } else {
    cache.oldest = &entry;
  }
  cache.newest = &entry;
  ++cache.ready_count;
}

void touch(ProgramCacheState &cache, ProgramCacheEntry &entry) noexcept {
  if (!linked(cache, entry) || cache.newest == &entry) {
    return;
  }
  unlink(cache, entry);
  append(cache, entry);
}

std::shared_ptr<ProgramCacheEntry> evict_ready(ProgramCacheState &cache) {
  if (cache.ready_count <= cache.capacity || cache.oldest == nullptr) {
    return {};
  }
  ProgramCacheEntry *const victim = cache.oldest;
  const auto slot = victim->slot;
  auto retired = std::move(slot->second);
  unlink(cache, *victim);
  cache.entries.erase(slot);
  ::rund::detail::counter::Accumulate(cache.evictions, 1u);
  return retired;
}

Result<std::shared_ptr<ProgramState>>
reuse(ProgramCacheState &cache, std::shared_ptr<ProgramCacheEntry> entry,
      std::unique_lock<std::mutex> &lock) {
  if (std::holds_alternative<ProgramCachePending>(entry->outcome)) {
    ::rund::detail::counter::Accumulate(cache.waits, 1u);
    entry->ready.wait(lock, [&] {
      return !std::holds_alternative<ProgramCachePending>(entry->outcome);
    });
  } else {
    ::rund::detail::counter::Accumulate(cache.hits, 1u);
  }
  if (const auto *const program =
          std::get_if<std::shared_ptr<ProgramState>>(&entry->outcome);
      program != nullptr) {
    touch(cache, *entry);
    return Result<std::shared_ptr<ProgramState>>::success(*program);
  }
  const Status &failure = std::get<Status>(entry->outcome);
  return Result<std::shared_ptr<ProgramState>>::fail(failure.reason());
}

} // namespace

Result<std::shared_ptr<ProgramState>>
cached_program(const std::shared_ptr<ProgramCacheState> &cache,
               const graph::Fingerprint fingerprint, ProgramBuilder builder) {
  if (cache == nullptr || cache->device == nullptr || cache->capacity == 0u) {
    return Result<std::shared_ptr<ProgramState>>::fail(
        Reason::ProgramCacheInvalid);
  }

  std::shared_ptr<ProgramCacheEntry> entry;
  {
    std::unique_lock lock{cache->mutex};
    const auto found = cache->entries.find(fingerprint);
    if (found != cache->entries.end()) {
      return reuse(*cache, found->second, lock);
    }
    try {
      entry = std::make_shared<ProgramCacheEntry>();
      const auto inserted = cache->entries.emplace(fingerprint, entry);
      entry->slot = inserted.first;
    } catch (const std::bad_alloc &) {
      return Result<std::shared_ptr<ProgramState>>::fail(
          Reason::ProgramCacheCapacity);
    }
    ::rund::detail::counter::Accumulate(cache->misses, 1u);
  }

  Result<std::shared_ptr<ProgramState>> built =
      Result<std::shared_ptr<ProgramState>>::fail(
          Reason::ProgramCompileException);
  try {
    built = builder();
  } catch (const std::bad_alloc &) {
    built =
        Result<std::shared_ptr<ProgramState>>::fail(Reason::ProgramCapacity);
  } catch (...) {
    built = Result<std::shared_ptr<ProgramState>>::fail(
        Reason::ProgramCompileException);
  }
  if (built && built.value() == nullptr) {
    built = Result<std::shared_ptr<ProgramState>>::fail(
        Reason::ProgramCompileException);
  }
  std::shared_ptr<ProgramCacheEntry> retired;
  {
    std::lock_guard lock{cache->mutex};
    if (built) {
      entry->outcome = built.value();
      append(*cache, *entry);
    } else {
      entry->outcome = Status::fail(built.reason());
      cache->entries.erase(entry->slot);
    }
    retired = evict_ready(*cache);
  }
  entry->ready.notify_all();
  retired.reset();
  return built;
}

void ProgramCacheState::clear_ready() noexcept {
  decltype(entries) retired;
  {
    std::lock_guard lock{mutex};
    for (auto entry = entries.begin(); entry != entries.end();) {
      if (std::holds_alternative<ProgramCachePending>(entry->second->outcome)) {
        ++entry;
        continue;
      }
      const auto ready = entry++;
      unlink(*this, *ready->second);
      retired.insert(retired.end(), entries.extract(ready));
    }
  }
}

} // namespace detail

Result<ProgramCache> program_cache(const Device &device,
                                   const std::size_t capacity) {
  if (device.state_ == nullptr) {
    return Result<ProgramCache>::fail(Reason::ProgramCacheInvalid);
  }
  if (capacity == 0u) {
    return Result<ProgramCache>::fail(Reason::ProgramCacheCapacity);
  }
  try {
    auto state = std::make_shared<detail::ProgramCacheState>();
    state->device = device.state_;
    state->capacity = capacity;
    return Result<ProgramCache>::success(ProgramCache{std::move(state)});
  } catch (const std::bad_alloc &) {
    return Result<ProgramCache>::fail(Reason::ProgramCacheCapacity);
  }
}

ProgramCache::Stats ProgramCache::stats() const noexcept {
  if (state_ == nullptr) {
    return {};
  }
  std::lock_guard lock{state_->mutex};
  Stats out{
      .hits = state_->hits,
      .misses = state_->misses,
      .waits = state_->waits,
      .evictions = state_->evictions,
      .ready_entries = state_->ready_count,
      .in_flight = state_->entries.size() - state_->ready_count,
      .capacity = state_->capacity,
  };
  return out;
}

void ProgramCache::clear() noexcept {
  if (state_ == nullptr) {
    return;
  }
  state_->clear_ready();
}

FlowBuilder on(const Device &device, const ProgramCache &cache) noexcept {
  if (device.state_ == nullptr || cache.state_ == nullptr ||
      cache.state_->device != device.state_) {
    return FlowBuilder{device.state_,
                       std::shared_ptr<detail::ProgramCacheState>{}};
  }
  return FlowBuilder{device.state_, cache.state_};
}

} // namespace rund::compute
