#include <rund/compute.hpp>

#include "../../allocation.hpp"
#include "src/compute/program/cache.hpp"
#include "src/compute/program/cache/state.hpp"

#include <condition_variable>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <new>
#include <thread>

namespace {

using Entry = rund::compute::detail::ProgramCacheEntry;
using State = rund::compute::detail::ProgramCacheState;
using Fingerprint = rund::compute::graph::Fingerprint;
using Cached =
    rund::compute::Result<std::shared_ptr<rund::compute::detail::ProgramState>>;

[[nodiscard]] std::shared_ptr<rund::compute::detail::ProgramState> program() {
  auto storage = std::make_shared<std::max_align_t>();
  return {storage, reinterpret_cast<rund::compute::detail::ProgramState *>(
                       storage.get())};
}

[[nodiscard]] std::shared_ptr<State> cache(const std::size_t capacity) {
  auto device = rund::compute::detail::open_cpu(1u);
  if (!device) {
    return {};
  }
  auto state = std::make_shared<State>();
  state->device = std::move(device).value();
  state->capacity = capacity;
  return state;
}

[[nodiscard]] bool lru_order_is_exact() {
  auto state = cache(2u);
  if (state == nullptr) {
    return false;
  }
  constexpr Fingerprint a{.hi = 1u, .lo = 1u};
  constexpr Fingerprint b{.hi = 1u, .lo = 2u};
  constexpr Fingerprint c{.hi = 1u, .lo = 3u};
  std::size_t builds = 0u;
  const auto build = [&] {
    ++builds;
    return Cached::success(program());
  };
  if (!rund::compute::detail::cached_program(state, a, build) ||
      !rund::compute::detail::cached_program(state, b, build) ||
      !rund::compute::detail::cached_program(state, a, build) ||
      !rund::compute::detail::cached_program(state, c, build)) {
    return false;
  }

  std::lock_guard lock{state->mutex};
  const auto a_slot = state->entries.find(a);
  const auto c_slot = state->entries.find(c);
  return builds == 3u && state->misses == 3u && state->hits == 1u &&
         state->evictions == 1u && state->ready_count == 2u &&
         state->entries.find(b) == state->entries.end() &&
         a_slot != state->entries.end() && c_slot != state->entries.end() &&
         state->oldest == a_slot->second.get() &&
         state->newest == c_slot->second.get() &&
         state->oldest->newer == state->newest &&
         state->newest->older == state->oldest;
}

[[nodiscard]] bool hit_allocates_nothing() {
  auto state = cache(1u);
  if (state == nullptr) {
    return false;
  }
  constexpr Fingerprint key{.hi = 3u, .lo = 1u};
  if (!rund::compute::detail::cached_program(
          state, key, [] { return Cached::success(program()); })) {
    return false;
  }
  bool built = false;
  node_compute_allocation::Start();
  const auto hit = rund::compute::detail::cached_program(state, key, [&] {
    built = true;
    return Cached::success(program());
  });
  node_compute_allocation::Stop();
  return hit && !built && node_compute_allocation::Count() == 0u;
}

[[nodiscard]] bool borrowed_builder_does_not_copy_or_allocate() {
  auto state = cache(1u);
  if (state == nullptr) { return false; }
  const auto owner = program();
  constexpr Fingerprint key{.hi = 9u, .lo = 1u};
  struct Builder final {
    std::array<std::uint64_t, 64u> payload{};
    std::shared_ptr<rund::compute::detail::ProgramState> owner;
    std::size_t calls{};
    explicit Builder(decltype(owner) value) : owner(std::move(value)) {}
    Builder(const Builder &) = delete;
    Builder(Builder &&) = delete;
    Cached operator()() { ++calls; ++payload[0]; return Cached::success(owner); }
  };
  Builder builder{owner};
  const auto miss = rund::compute::detail::cached_program(state, key, builder);
  if (!miss || miss.value() != owner || builder.calls != 1u) { return false; }
  node_compute_allocation::Start();
  node_compute_allocation::FailNext();
  const auto hit = rund::compute::detail::cached_program(state, key, builder);
  const auto temporary = rund::compute::detail::cached_program(
      state, key, Builder{owner});
  node_compute_allocation::ClearFailure();
  node_compute_allocation::Stop();
  if (!hit || !temporary || hit.value() != owner || temporary.value() != owner ||
      builder.calls != 1u || node_compute_allocation::Count() != 0u) {
    return false;
  }
  state->clear_ready();
  const auto fresh = rund::compute::detail::cached_program(
      state, key, [token = std::make_unique<unsigned>(7u), owner]() mutable {
        return *token == 7u ? Cached::success(owner)
                           : Cached::fail(rund::compute::Reason::ProgramCompileException);
      });
  if (!fresh || fresh.value() != owner) { return false; }
  state->clear_ready();
  const auto capacity = rund::compute::detail::cached_program(
      state, key, []() -> Cached { throw std::bad_alloc{}; });
  const auto exception = rund::compute::detail::cached_program(
      state, key, []() -> Cached { throw 7u; });
  return !capacity && !exception &&
         capacity.reason() == rund::compute::Reason::ProgramCapacity &&
         exception.reason() == rund::compute::Reason::ProgramCompileException &&
         state->entries.empty();
}

[[nodiscard]] bool allocation_failure_leaves_no_entry() {
  // Walk every allocation boundary through the first successful admission,
  // including index node and initial bucket allocation on this implementation.
  for (std::uint64_t allowed = 0u; allowed < 16u; ++allowed) {
    auto state = cache(1u);
    if (state == nullptr) { return false; }
    const auto owner = program();
    constexpr Fingerprint key{.hi = 4u, .lo = 1u};
    bool built = false;
    const auto build = [&] { built = true; return Cached::success(owner); };
    node_compute_allocation::FailAfter(allowed);
    const auto result = rund::compute::detail::cached_program(state, key, build);
    node_compute_allocation::ClearFailure();
    if (result) {
      return allowed != 0u && built && result.value() == owner &&
             state->misses == 1u && state->ready_count == 1u;
    }
    if (built || result.reason() != rund::compute::Reason::ProgramCacheCapacity ||
        !state->entries.empty() || state->misses != 0u || state->ready_count != 0u) {
      return false;
    }
    const auto retry = rund::compute::detail::cached_program(state, key, build);
    if (!retry || retry.value() != owner) { return false; }
  }
  return false;
}

[[nodiscard]] bool clear_keeps_pending() {
  auto state = cache(2u);
  if (state == nullptr) {
    return false;
  }
  constexpr Fingerprint ready{.hi = 2u, .lo = 1u};
  constexpr Fingerprint pending{.hi = 2u, .lo = 2u};
  if (!rund::compute::detail::cached_program(
          state, ready, [] { return Cached::success(program()); })) {
    return false;
  }

  std::mutex gate;
  std::condition_variable changed;
  bool entered = false;
  bool release = false;
  std::optional<Cached> result;
  std::thread builder{[&] {
    result.emplace(rund::compute::detail::cached_program(state, pending, [&] {
      std::unique_lock lock{gate};
      entered = true;
      changed.notify_one();
      changed.wait(lock, [&] { return release; });
      return Cached::success(program());
    }));
  }};
  {
    std::unique_lock lock{gate};
    changed.wait(lock, [&] { return entered; });
  }

  state->clear_ready();
  bool exact = false;
  {
    std::lock_guard lock{state->mutex};
    const auto slot = state->entries.find(pending);
    exact = state->entries.size() == 1u && state->ready_count == 0u &&
            state->oldest == nullptr && state->newest == nullptr &&
            slot != state->entries.end() &&
            std::holds_alternative<rund::compute::detail::ProgramCachePending>(
                slot->second->outcome);
  }
  {
    std::lock_guard lock{gate};
    release = true;
  }
  changed.notify_one();
  builder.join();
  {
    std::lock_guard lock{state->mutex};
    exact = exact && result && *result && state->entries.size() == 1u &&
            state->ready_count == 1u && state->oldest == state->newest &&
            state->oldest != nullptr;
  }
  return exact;
}

[[nodiscard]] bool collisions_preserve_full_identity() {
  constexpr std::size_t count = 64u;
  auto state = cache(count);
  if (state == nullptr) { return false; }
  std::array<std::shared_ptr<rund::compute::detail::ProgramState>, count> owners;
  std::array<Fingerprint, count> keys;
  const rund::compute::detail::FingerprintHash hash;
  for (std::size_t i = 0; i < count; ++i) {
    keys[i] = {.hi = i, .lo = i - 0x9e3779b97f4a7c15ull};
    if (hash(keys[i]) != 0u) { return false; }
    owners[i] = program();
    const auto result = rund::compute::detail::cached_program(
        state, keys[i], [&] { return Cached::success(owners[i]); });
    if (!result || result.value() != owners[i]) { return false; }
  }
  // Growth rehashes membership; neither identity nor LRU stores an iterator.
  for (std::size_t i = count; i-- > 0u;) {
    const auto result = rund::compute::detail::cached_program(
        state, keys[i], [] { return Cached::fail(rund::compute::Reason::ProgramCompileException); });
    if (!result || result.value() != owners[i]) { return false; }
  }
  node_compute_allocation::Start();
  state->clear_ready();
  node_compute_allocation::Stop();
  return node_compute_allocation::Count() == 0u && state->entries.empty() &&
         state->ready_count == 0u && state->oldest == nullptr &&
         state->newest == nullptr && state->misses == count && state->hits == count;
}

} // namespace

int RunComputeProgramCacheIndexContract() {
  if (!lru_order_is_exact()) {
    return 1;
  }
  if (!hit_allocates_nothing()) {
    return 2;
  }
  if (!allocation_failure_leaves_no_entry()) {
    return 3;
  }
  if (!clear_keeps_pending()) {
    return 4;
  }
  if (!collisions_preserve_full_identity()) { return 5; }
  if (!borrowed_builder_does_not_copy_or_allocate()) { return 6; }
  return 0;
}
