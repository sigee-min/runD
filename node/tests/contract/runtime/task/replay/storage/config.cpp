#include "test/assert.hpp"

#include <rund/replay.hpp>
#include <rund/session.hpp>

#include <string_view>

namespace {

[[nodiscard]] rund::SessionConfig ValidRuntimeOptions() {
  rund::SessionConfig options{};
  options.id = 1u;
  options.workers = 1u;
  return options;
}

[[nodiscard]] rund::Session::Status
Configure(const ::rund::replay::Storage &storage) {
  rund::Session runtime{};
  rund::SessionConfig options = ValidRuntimeOptions();
  options.replay.storage = storage;
  return runtime.open(options);
}

} // namespace

int RunRuntimeTaskReplayStorageConfigContract() {
  const ::rund::replay::Storage defaults{};
  TEST_ASSERT(defaults.mode == ::rund::replay::StorageMode::Memory);
  TEST_ASSERT(defaults.directory.empty());
  TEST_ASSERT(defaults.max_allocated_bytes ==
              2ull * 1024ull * 1024ull * 1024ull);
  TEST_ASSERT(defaults.minimum_free_bytes == 0u);
  TEST_ASSERT(!defaults.budget);

  ::rund::replay::Storage memory_only{};
  memory_only.mode = ::rund::replay::StorageMode::Memory;
  memory_only.directory.clear();
  TEST_ASSERT(Configure(memory_only));

  ::rund::replay::Storage memory_with_spill{};
  memory_with_spill.mode = ::rund::replay::StorageMode::Memory;
  memory_with_spill.directory = ".cache/host-replay-spill";
  const rund::Session::Status memory_with_spill_decision =
      Configure(memory_with_spill);
  TEST_ASSERT(!memory_with_spill_decision);
  TEST_ASSERT(memory_with_spill_decision.code() ==
              rund::ReasonCode::HostReplayStorageInvalid);

  ::rund::replay::Storage spill_missing_directory{};
  spill_missing_directory.mode = ::rund::replay::StorageMode::Spill;
  spill_missing_directory.directory.clear();
  const rund::Session::Status missing_directory_decision =
      Configure(spill_missing_directory);
  TEST_ASSERT(!missing_directory_decision);
  TEST_ASSERT(missing_directory_decision.code() ==
              rund::ReasonCode::HostReplayStorageInvalid);

  ::rund::replay::Storage zero_memory_cache{};
  zero_memory_cache.cached_bytes = 0u;
  TEST_ASSERT(Configure(zero_memory_cache));

  ::rund::replay::Storage zero_segment{};
  zero_segment.segment_bytes = 0u;
  TEST_ASSERT(Configure(zero_segment));

  ::rund::replay::Storage quota_below_segment{};
  quota_below_segment.segment_bytes = 4096u;
  quota_below_segment.max_bytes = 4095u;
  TEST_ASSERT(Configure(quota_below_segment));

  ::rund::replay::Storage unknown_mode{};
  unknown_mode.mode = static_cast<::rund::replay::StorageMode>(255u);
  const rund::Session::Status unknown_mode_decision = Configure(unknown_mode);
  TEST_ASSERT(!unknown_mode_decision);
  TEST_ASSERT(unknown_mode_decision.code() ==
              rund::ReasonCode::HostReplayStorageInvalid);

  ::rund::replay::Storage spill{};
  spill.mode = ::rund::replay::StorageMode::Spill;
  spill.directory = ".cache/host-replay-spill";
  spill.cached_bytes = 4096u;
  spill.segment_bytes = 4096u;
  spill.max_bytes = 8192u;
  TEST_ASSERT(Configure(spill));

  ::rund::replay::Storage spill_without_cache = spill;
  spill_without_cache.cached_bytes = 0u;
  TEST_ASSERT(Configure(spill_without_cache));

  ::rund::replay::Storage spill_without_segment = spill;
  spill_without_segment.segment_bytes = 0u;
  const rund::Session::Status spill_without_segment_decision =
      Configure(spill_without_segment);
  TEST_ASSERT(!spill_without_segment_decision);
  TEST_ASSERT(spill_without_segment_decision.code() ==
              rund::ReasonCode::HostReplayStorageInvalid);

  ::rund::replay::Storage spill_without_allocation = spill;
  spill_without_allocation.max_allocated_bytes = 0u;
  const rund::Session::Status spill_without_allocation_decision =
      Configure(spill_without_allocation);
  TEST_ASSERT(!spill_without_allocation_decision);
  TEST_ASSERT(spill_without_allocation_decision.code() ==
              rund::ReasonCode::HostReplayStorageInvalid);

  rund::SessionConfig spec{};
  spec.replay.storage = spill;
  const rund::Session::Result result = rund::run(spec, [] {});
  TEST_ASSERT(result.ok());

  return 0;
}
