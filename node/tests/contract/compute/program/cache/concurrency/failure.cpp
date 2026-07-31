#include "model.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <optional>
#include <thread>

namespace node_compute_cache_contract {
namespace {

[[nodiscard]] bool in_flight_failure_is_shared() {
  auto device = rund::compute::detail::open_cpu(1u);
  if (!device) {
    return false;
  }
  auto cache = std::make_shared<rund::compute::detail::ProgramCacheState>();
  cache->device = std::move(device).value();
  cache->capacity = 1u;
  const rund::compute::graph::Fingerprint key{.hi = 0x91a2u, .lo = 0xb3c4u};

  std::mutex builder_mutex;
  std::condition_variable builder_ready;
  bool builder_entered = false;
  bool release_builder = false;
  std::atomic<std::uint32_t> build_count{0u};
  const auto blocked_failure = [&] {
    build_count.fetch_add(1u, std::memory_order_relaxed);
    std::unique_lock lock{builder_mutex};
    builder_entered = true;
    builder_ready.notify_one();
    builder_ready.wait(lock, [&] { return release_builder; });
    return CachedResult::fail(rund::compute::Reason::ProgramCompileException);
  };

  std::optional<CachedResult> compiler_result;
  std::optional<CachedResult> waiter_result;
  std::thread compiler{[&] {
    compiler_result.emplace(
        rund::compute::detail::cached_program(cache, key, blocked_failure));
  }};
  {
    std::unique_lock lock{builder_mutex};
    builder_ready.wait(lock, [&] { return builder_entered; });
  }
  std::thread waiter{[&] {
    waiter_result.emplace(
        rund::compute::detail::cached_program(cache, key, blocked_failure));
  }};

  bool waiter_is_blocked = false;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds{5};
  while (std::chrono::steady_clock::now() < deadline) {
    {
      std::lock_guard lock{cache->mutex};
      waiter_is_blocked = cache->waits == 1u;
    }
    if (waiter_is_blocked) {
      break;
    }
    std::this_thread::yield();
  }
  {
    std::lock_guard lock{builder_mutex};
    release_builder = true;
  }
  builder_ready.notify_one();
  compiler.join();
  waiter.join();

  return waiter_is_blocked &&
         build_count.load(std::memory_order_relaxed) == 1u && compiler_result &&
         waiter_result && !*compiler_result && !*waiter_result &&
         compiler_result->reason() ==
             rund::compute::Reason::ProgramCompileException &&
         waiter_result->reason() ==
             rund::compute::Reason::ProgramCompileException &&
         cache->misses == 1u && cache->waits == 1u && cache->entries.empty();
}

[[nodiscard]] bool failed_build_is_not_cached() {
  auto device = rund::compute::detail::open_cpu(1u);
  if (!device) {
    return false;
  }
  auto cache = std::make_shared<rund::compute::detail::ProgramCacheState>();
  cache->device = std::move(device).value();
  cache->capacity = 1u;
  const rund::compute::graph::Fingerprint key{.hi = 0x1234u, .lo = 0x5678u};
  const auto fail = [] {
    return CachedResult::fail(rund::compute::Reason::ProgramCompileException);
  };
  const auto first = rund::compute::detail::cached_program(cache, key, fail);
  const auto retry = rund::compute::detail::cached_program(cache, key, fail);
  return !first && !retry &&
         first.reason() == rund::compute::Reason::ProgramCompileException &&
         retry.reason() == rund::compute::Reason::ProgramCompileException &&
         cache->misses == 2u && cache->hits == 0u && cache->waits == 0u &&
         cache->entries.empty();
}

[[nodiscard]] bool invalid_builder_outcomes_are_not_cached() {
  auto device = rund::compute::detail::open_cpu(1u);
  if (!device) {
    return false;
  }
  auto cache = std::make_shared<rund::compute::detail::ProgramCacheState>();
  cache->device = std::move(device).value();
  cache->capacity = 1u;
  const rund::compute::graph::Fingerprint null_key{.hi = 0xa11u, .lo = 0u};
  const auto null_program = [] {
    return CachedResult::success(
        std::shared_ptr<rund::compute::detail::ProgramState>{});
  };
  const auto first_null =
      rund::compute::detail::cached_program(cache, null_key, null_program);
  const auto second_null =
      rund::compute::detail::cached_program(cache, null_key, null_program);

  const rund::compute::graph::Fingerprint ok_failure_key{.hi = 0xa11u,
                                                         .lo = 1u};
  const auto forged_ok =
      rund::compute::detail::cached_program(cache, ok_failure_key, [] {
        return CachedResult::fail(static_cast<rund::compute::Reason>(0xffffu));
      });
  return !first_null && !second_null && !forged_ok &&
         first_null.code() == rund::compute::Code::Compile &&
         second_null.code() == rund::compute::Code::Compile &&
         first_null.error() == "compute_program_compile_exception" &&
         second_null.error() == "compute_program_compile_exception" &&
         forged_ok.code() == rund::compute::Code::Invalid &&
         forged_ok.reason() == rund::compute::Reason::ReasonInvalid &&
         forged_ok.error() == "compute_reason_invalid" && cache->misses == 3u &&
         cache->entries.empty();
}

} // namespace

int RunFailure() {
  if (!in_flight_failure_is_shared()) {
    return 14;
  }
  if (!failed_build_is_not_cached()) {
    return 15;
  }
  if (!invalid_builder_outcomes_are_not_cached()) {
    return 16;
  }
  return 0;
}

} // namespace node_compute_cache_contract
