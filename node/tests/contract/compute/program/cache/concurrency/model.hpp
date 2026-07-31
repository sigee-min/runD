#pragma once

#include "src/compute/program/cache.hpp"
#include <rund/compute.hpp>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <variant>

namespace node_compute_cache_contract {

using rund::compute::Target;

using CacheEntries =
    decltype(rund::compute::detail::ProgramCacheState::entries);
static_assert(std::is_same_v<typename CacheEntries::key_type,
                             rund::compute::graph::Fingerprint>);
static_assert(std::is_empty_v<rund::compute::detail::FingerprintLess>);
static_assert(!std::is_final_v<rund::compute::detail::FingerprintLess>);
static_assert(sizeof(rund::compute::graph::Fingerprint) ==
              2u * sizeof(std::uint64_t));
static_assert(std::variant_size_v<rund::compute::detail::ProgramCacheOutcome> ==
              3u);
static_assert(
    std::is_same_v<std::variant_alternative_t<
                       0u, rund::compute::detail::ProgramCacheOutcome>,
                   rund::compute::detail::ProgramCachePending>);
static_assert(
    std::is_same_v<std::variant_alternative_t<
                       1u, rund::compute::detail::ProgramCacheOutcome>,
                   std::shared_ptr<rund::compute::detail::ProgramState>>);
static_assert(
    std::is_same_v<std::variant_alternative_t<
                       2u, rund::compute::detail::ProgramCacheOutcome>,
                   rund::compute::Status>);

using CachedResult =
    rund::compute::Result<std::shared_ptr<rund::compute::detail::ProgramState>>;

[[nodiscard]] inline std::shared_ptr<rund::compute::detail::ProgramState>
opaque_program_owner() {
  auto storage = std::make_shared<std::max_align_t>();
  return {storage, reinterpret_cast<rund::compute::detail::ProgramState *>(
                       storage.get())};
}

struct DestructionGate final {
  std::mutex mutex;
  std::condition_variable changed;
  bool entered = false;
  bool released = false;
};

struct BlockingProgramStorage final {
  explicit BlockingProgramStorage(std::shared_ptr<DestructionGate> gate_in)
      : gate(std::move(gate_in)) {}

  ~BlockingProgramStorage() {
    std::unique_lock lock{gate->mutex};
    gate->entered = true;
    gate->changed.notify_one();
    gate->changed.wait(lock, [&] { return gate->released; });
  }

  std::shared_ptr<DestructionGate> gate;
};

[[nodiscard]] inline std::shared_ptr<rund::compute::detail::ProgramState>
blocking_program_owner(const std::shared_ptr<DestructionGate> &gate) {
  auto storage = std::make_shared<BlockingProgramStorage>(gate);
  return {storage, reinterpret_cast<rund::compute::detail::ProgramState *>(
                       storage.get())};
}

} // namespace node_compute_cache_contract
