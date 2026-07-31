#include "test/assert.hpp"

#include "../coroutine/allocation.hpp"
#include "local.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <utility>

int CheckReplayFailureContract() {
  rund::Session session{};
  rund::SessionConfig config{};
  config.workers = 1u;
  TEST_ASSERT(session.open(config));

  const rund::replay::Record expected =
      rund::replay::record(session, [](rund::replay::Context &) noexcept {});
  TEST_ASSERT(expected);

  rund::Session::Result record_result = session.scope([] {});
  TEST_ASSERT(record_result);
  runtime_task_allocation::FailNext();
  const rund::replay::Record record_failure =
      rund::replay::detail::build_record(session, std::move(record_result),
                                         rund::replay::detail::genesis_start());
  TEST_ASSERT(record_failure.code() == rund::replay::Code::AllocationFailed);

  rund::Session::Result check_result = session.scope([] {});
  TEST_ASSERT(check_result);
  runtime_task_allocation::FailNext();
  const rund::replay::Check check_failure = rund::replay::detail::check_result(
      expected, session, std::move(check_result), expected.start_hash());
  TEST_ASSERT(check_failure.code() == rund::replay::Code::AllocationFailed);

  runtime_task_allocation::FailNext();
  const rund::replay::Diff diff_failure =
      rund::replay::diff(expected, expected);
  TEST_ASSERT(diff_failure.code() == rund::replay::Code::AllocationFailed);

  runtime_task_allocation::FailNext();
  const rund::replay::Window window_failure =
      rund::replay::window(expected, expected);
  TEST_ASSERT(window_failure.code() == rund::replay::Code::AllocationFailed);

  constexpr std::uint64_t schema = 0x73746174652d7631ull;
  const std::array state_bytes{std::byte{0x41}, std::byte{0x42}};
  auto reject_restore = [](std::span<const std::byte>) {
    return rund::replay::Restore::Failed;
  };
  const rund::replay::Binding rejected_binding{schema, reject_restore};
  runtime_task_allocation::FailNext();
  const rund::replay::Checkpoint checkpoint_failure =
      rejected_binding.checkpoint(expected, state_bytes);
  TEST_ASSERT(checkpoint_failure.code() ==
              rund::replay::Code::AllocationFailed);

  const rund::replay::Checkpoint saved =
      rejected_binding.checkpoint(expected, state_bytes);
  TEST_ASSERT(saved);

  bool rejected_callback = false;
  const auto rejected_resume = rejected_binding.resume(saved);
  TEST_ASSERT(rejected_resume);
  const rund::replay::Record rejected_restore = rejected_resume.record(
      session, [&](rund::replay::Context &) { rejected_callback = true; });
  TEST_ASSERT(!rejected_restore);
  TEST_ASSERT(rejected_restore.code() ==
              rund::replay::Code::CheckpointRestoreFailed);
  TEST_ASSERT(!rejected_callback);

  auto throw_restore = [](std::span<const std::byte>) -> rund::replay::Restore {
    throw 1;
  };
  const rund::replay::Binding throwing_binding{schema, throw_restore};
  const auto throwing_resume = throwing_binding.resume(saved);
  TEST_ASSERT(throwing_resume);
  const rund::replay::Record threw = throwing_resume.record(
      session, [&](rund::replay::Context &) { rejected_callback = true; });
  TEST_ASSERT(!threw);
  TEST_ASSERT(threw.code() == rund::replay::Code::CheckpointRestoreThrew);
  TEST_ASSERT(!rejected_callback);

  const auto restore =
      [state_bytes](const std::span<const std::byte> restored) {
        return restored.size() == state_bytes.size()
                   ? rund::replay::Restore::Restored
                   : rund::replay::Restore::Failed;
      };
  const rund::replay::Binding binding{schema, restore};
  const auto resume = binding.resume(saved);
  TEST_ASSERT(resume);
  const rund::replay::Record continuation =
      resume.record(session, [](rund::replay::Context &) noexcept {});
  TEST_ASSERT(continuation);
  runtime_task_allocation::FailNext();
  const rund::replay::Checkpoint advance_failure =
      binding.advance(saved, continuation, state_bytes);
  TEST_ASSERT(advance_failure.code() == rund::replay::Code::AllocationFailed);

  runtime_task_allocation::FailNext();
  const rund::replay::Scenario scenario_failure = rund::replay::scenario(
      session, expected, std::span<const rund::replay::Choice>{},
      [](rund::replay::Context &) noexcept {});
  TEST_ASSERT(scenario_failure.code() == rund::replay::Code::AllocationFailed);

  TEST_ASSERT(session.close());
  return 0;
}
