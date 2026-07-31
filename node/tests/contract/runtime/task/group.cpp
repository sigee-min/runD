#include "coroutine/allocation.hpp"
#include "test/assert.hpp"

#include <rund/session.hpp>
#include <rund/task/channel.hpp>
#include <rund/task/group.hpp>

#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>

static_assert(
    !std::is_copy_constructible_v<rund::task::Group::JoinState>);
static_assert(!std::is_copy_assignable_v<rund::task::Group::JoinState>);
static_assert(
    std::is_nothrow_move_constructible_v<rund::task::Group::JoinState>);
static_assert(!std::is_move_assignable_v<rund::task::Group::JoinState>);
static_assert(!std::is_copy_constructible_v<rund::task::Group>);
static_assert(!std::is_copy_assignable_v<rund::task::Group>);
static_assert(!std::is_move_constructible_v<rund::task::Group>);
static_assert(!std::is_move_assignable_v<rund::task::Group>);
static_assert(sizeof(rund::task::Group) == 32u);
static_assert(sizeof(rund::task::Group::JoinState) == 24u);

namespace {

rund::task::Task<void> RunGroup(const std::span<rund::task::Handle> slots,
                                const std::span<std::uint64_t> values,
                                bool *const handles_valid,
                                rund::task::Status *const status) {
  rund::task::Group group{slots};
  for (std::size_t index = 0u; index < values.size(); ++index) {
    const rund::task::Handle handle =
        group.spawn("task-group-worker", [values, index] {
          values[index] = static_cast<std::uint64_t>(index + 1u);
        });
    *handles_valid = *handles_valid && handle;
  }
  const rund::task::Status joined = co_await group.join();
  *status = joined;
}

rund::task::Task<void> FailTask() {
  throw 1;
  co_return;
}

rund::task::Task<void>
RunFailingGroup(const std::span<rund::task::Handle> slots,
                rund::task::Status *const status) {
  rund::task::Group group{slots};
  if (!group.spawn("task-group-failure", FailTask()) ||
      !group.spawn("task-group-after-failure", [] {})) {
    *status = rund::task::Status::fail(
        rund::ReasonCode::TaskCapacityExceeded);
    co_return;
  }
  *status = co_await group.join();
}

rund::task::Task<void> RunSmallGroup(
    const std::span<rund::task::Handle> slots, std::uint64_t *const value,
    bool *const handles_valid, bool *const size_is_two, bool *const nonempty,
    bool *const empty_after_join, bool *const moved_from_inert,
    bool *const concurrent_join_rejected,
    rund::task::Status *const status) {
  rund::task::Group tasks{slots};
  const rund::task::Handle one = tasks.spawn("one", [value] { *value += 1u; });
  const rund::task::Handle two = tasks.spawn("two", [value] { *value += 2u; });
  *handles_valid = one && two;
  *size_is_two = tasks.size() == 2u;
  *nonempty = !tasks.empty();
  auto original = tasks.begin_join();
  auto duplicate = tasks.begin_join();
  *concurrent_join_rejected =
      duplicate.finish().code() == rund::ReasonCode::TaskInvalid &&
      !tasks.spawn("join-active-spawn", [] {}) && tasks.size() == 2u;
  tasks.clear();
  *concurrent_join_rejected =
      *concurrent_join_rejected && tasks.size() == 2u;
  auto joining = std::move(original);
  *moved_from_inert = !original.pending() && !original.current() &&
                       original.finish().code() ==
                           rund::ReasonCode::TaskInvalid;
  while (joining.pending()) {
    joining.advance(co_await joining.current());
  }
  *status = joining.finish();
  *empty_after_join = tasks.empty();
}

rund::task::Task<void> HoldSharedJoinTarget(
    std::atomic<std::uint32_t> *const admitted,
    std::atomic<std::uint32_t> *const resumed,
    const std::uint32_t expected_waiters, bool *const saturated) {
  const rund::task::Status yielded = co_await rund::task::yield();
  *saturated = yielded &&
               admitted->load(std::memory_order_relaxed) == expected_waiters &&
               resumed->load(std::memory_order_relaxed) == 0u;
}

rund::task::Task<void> AwaitSharedJoinTarget(
    const rund::task::Handle target, const std::uint32_t index,
    std::atomic<std::uint32_t> *const admitted,
    std::atomic<std::uint32_t> *const resumed,
    const std::span<rund::task::Status> results,
    const std::span<std::uint32_t> wake_order) {
  admitted->fetch_add(1u, std::memory_order_relaxed);
  results[index] = co_await target;
  const std::uint32_t position =
      resumed->fetch_add(1u, std::memory_order_relaxed);
  if (position < wake_order.size()) {
    wake_order[position] = index;
  }
}

rund::task::Task<void> WaitJoinGate(
    rund::task::channel<int> *const gate) {
  static_cast<void>(co_await gate->recv());
}

rund::task::Task<void> CompleteJoinBarrier() {
  co_return;
}

rund::task::Task<void> RunMixedJoinOrder(
    const std::span<rund::task::Status> results,
    const std::span<std::uint32_t> wake_order,
    std::atomic<std::uint32_t> *const admitted,
    std::atomic<std::uint32_t> *const resumed, bool *const handles_valid,
    rund::task::Status *const targets_joined,
    rund::task::Status *const waiters_joined) {
  auto first_gate = rund::task::channel<int>::make(0u);
  auto second_gate = rund::task::channel<int>::make(0u);
  const rund::task::Handle first =
      rund::task::spawn("mixed-join-first", WaitJoinGate(&first_gate));
  const rund::task::Handle second =
      rund::task::spawn("mixed-join-second", WaitJoinGate(&second_gate));
  std::array<rund::task::Handle, 4u> waiters{};
  const std::array<rund::task::Handle, 4u> targets{
      first,
      second,
      first,
      second,
  };
  *handles_valid = first && second;
  for (std::uint32_t index = 0u; index < waiters.size(); ++index) {
    waiters[index] = rund::task::spawn(
        "mixed-join-waiter",
        AwaitSharedJoinTarget(targets[index], index, admitted, resumed,
                              results, wake_order));
    *handles_valid = *handles_valid && waiters[index];
  }

  const rund::task::Handle barrier =
      rund::task::spawn("mixed-join-barrier", CompleteJoinBarrier());
  *handles_valid = *handles_valid && barrier;
  static_cast<void>(co_await barrier);
  const rund::task::Status first_sent = co_await first_gate.send(1);
  const rund::task::Status first_joined = co_await first;
  const rund::task::Status second_sent = co_await second_gate.send(1);
  const rund::task::Status second_joined = co_await second;
  *targets_joined = first_sent && first_joined && second_sent && second_joined
                        ? rund::task::Status::success()
                        : rund::task::Status::fail(
                              !first_sent    ? first_sent.code()
                              : !first_joined ? first_joined.code()
                              : !second_sent ? second_sent.code()
                                             : second_joined.code());
  *waiters_joined = rund::task::Status::success();
  for (const rund::task::Handle waiter : waiters) {
    const rund::task::Status joined = co_await waiter;
    if (!joined && *waiters_joined) {
      *waiters_joined = joined;
    }
  }
}

} // namespace

int RunRuntimeTaskTaskGroupContract() {
  constexpr std::size_t kTasks = 32u;
  std::array<std::uint64_t, kTasks> values{};
  std::array<rund::task::Handle, kTasks> slots{};
  bool all_handles_valid = true;
  rund::task::Status group_status = rund::task::Status::success();
  rund::task::Status coordinator_joined{};

  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
        .workers = 4u,
        .scheduler = {
          .task_capacity = 64u,
          .ready_queue_capacity = 64u,
        },
      },
      [&] {
        const rund::task::Handle coordinator = rund::task::spawn(
            "task-group-coordinator",
            RunGroup(slots, values, &all_handles_valid, &group_status));
        coordinator_joined = rund::task::join(coordinator);
      });

  TEST_ASSERT(report.ok());
  TEST_ASSERT(all_handles_valid);
  TEST_ASSERT(coordinator_joined.ok());
  TEST_ASSERT(group_status.ok());
  for (std::size_t index = 0u; index < kTasks; ++index) {
    TEST_ASSERT(values[index] == index + 1u);
  }

  std::array<rund::task::Handle, 2u> failure_slots{};
  rund::task::Status failure_status{};
  rund::task::Status failure_coordinator{};
  const rund::Session::Result failure_report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler = {
              .task_capacity = 8u,
              .ready_queue_capacity = 8u,
          },
      },
      [&] {
        failure_coordinator = rund::task::join(rund::task::spawn(
            "task-group-failure-owner",
            RunFailingGroup(failure_slots, &failure_status)));
      });
  TEST_ASSERT(failure_report.ok());
  TEST_ASSERT(failure_coordinator.ok());
  TEST_ASSERT(failure_status.code() == rund::ReasonCode::TaskFailed);
  TEST_ASSERT(!failure_status.error().empty());
  TEST_ASSERT(failure_status.exit_code() == 1);

  std::uint64_t value = 0u;
  std::array<rund::task::Handle, 2u> small_slots{};
  bool size_is_two = false;
  bool nonempty_after_spawn = false;
  bool ergonomic_handles_valid = false;
  bool empty_after_join = false;
  bool moved_from_inert = false;
  bool concurrent_join_rejected = false;
  rund::task::Status small_status = rund::task::Status::success();
  rund::task::Status small_joined{};
  const rund::Session::Result ergonomics_report = rund::run(
      rund::SessionConfig{
        .workers = 1u,
        .scheduler = {
          .task_capacity = 8u,
          .ready_queue_capacity = 8u,
        },
      },
      [&] {
        const rund::task::Handle coordinator = rund::task::spawn(
            "small-task-group",
            RunSmallGroup(small_slots, &value, &ergonomic_handles_valid,
                          &size_is_two, &nonempty_after_spawn,
                          &empty_after_join, &moved_from_inert,
                          &concurrent_join_rejected, &small_status));
        small_joined = rund::task::join(coordinator);
      });

  TEST_ASSERT(ergonomics_report.ok());
  TEST_ASSERT(small_joined.ok());
  TEST_ASSERT(ergonomic_handles_valid);
  TEST_ASSERT(size_is_two);
  TEST_ASSERT(nonempty_after_spawn);
  TEST_ASSERT(small_status.ok());
  TEST_ASSERT(empty_after_join);
  TEST_ASSERT(moved_from_inert);
  TEST_ASSERT(concurrent_join_rejected);
  TEST_ASSERT(value == 3u);

  std::array<rund::task::Status, 4u> mixed_results{};
  std::array<std::uint32_t, 4u> mixed_wake_order{};
  std::atomic<std::uint32_t> mixed_admitted{0u};
  std::atomic<std::uint32_t> mixed_resumed{0u};
  bool mixed_handles_valid = false;
  rund::task::Status mixed_targets_joined{};
  rund::task::Status mixed_waiters_joined{};
  rund::task::Status mixed_coordinator_joined{};
  const rund::Session::Result mixed_report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler = {
              .task_workers = 1u,
              .task_capacity = 8u,
              .ready_queue_capacity = 8u,
              .channel_capacity = 2u,
              .channel_wait_capacity = 2u,
          },
      },
      [&] {
        const rund::task::Handle coordinator = rund::task::spawn(
            "mixed-join-order",
            RunMixedJoinOrder(mixed_results, mixed_wake_order,
                              &mixed_admitted, &mixed_resumed,
                              &mixed_handles_valid, &mixed_targets_joined,
                              &mixed_waiters_joined));
        mixed_coordinator_joined = rund::task::join(coordinator);
      });

  TEST_ASSERT(mixed_report.ok());
  TEST_ASSERT(mixed_handles_valid);
  TEST_ASSERT(mixed_coordinator_joined.ok());
  TEST_ASSERT(mixed_targets_joined.ok());
  TEST_ASSERT(mixed_waiters_joined.ok());
  TEST_ASSERT(mixed_admitted.load(std::memory_order_relaxed) == 4u);
  TEST_ASSERT(mixed_resumed.load(std::memory_order_relaxed) == 4u);
  for (const rund::task::Status result : mixed_results) {
    TEST_ASSERT(result.ok());
  }
  constexpr std::array<std::uint32_t, 4u> kMixedCanonicalOrder{0u, 2u, 1u,
                                                               3u};
  TEST_ASSERT(mixed_wake_order == kMixedCanonicalOrder);

  constexpr std::uint32_t kJoinCapacity = 8u;
  constexpr std::uint32_t kJoinWaiters = kJoinCapacity - 1u;
  static_assert(kJoinWaiters > 1u);
  std::array<rund::task::Handle, kJoinWaiters> waiter_handles{};
  std::array<rund::task::Status, kJoinWaiters> waiter_results{};
  std::array<rund::task::Status, kJoinWaiters> waiter_terminals{};
  std::array<std::uint32_t, kJoinWaiters> wake_order{};
  std::atomic<std::uint32_t> admitted{0u};
  std::atomic<std::uint32_t> resumed{0u};
  bool saturation_seen = false;
  bool saturation_handles_valid = true;
  rund::task::Status target_terminal{};
  std::uint64_t join_allocations =
      std::numeric_limits<std::uint64_t>::max();

  const rund::Session::Result saturation_report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler = {
              .task_workers = 1u,
              .task_capacity = kJoinCapacity,
              .ready_queue_capacity = kJoinCapacity,
          },
      },
      [&] {
        // Prime frame-size classes and scheduler dispatch without admitting a
        // valid join edge. The measured pass therefore isolates join-wait
        // storage rather than cold coroutine-frame pages.
        const rund::task::Handle prime_target = rund::task::spawn(
            "join-target-prime",
            HoldSharedJoinTarget(&admitted, &resumed, kJoinWaiters,
                                 &saturation_seen));
        TEST_ASSERT(prime_target);
        for (std::uint32_t index = 0u; index < kJoinWaiters; ++index) {
          waiter_handles[index] = rund::task::spawn(
              "join-wait-prime",
              AwaitSharedJoinTarget({}, index, &admitted, &resumed,
                                    waiter_results, wake_order));
          TEST_ASSERT(waiter_handles[index]);
        }
        TEST_ASSERT(rund::task::join(prime_target));
        for (const rund::task::Handle handle : waiter_handles) {
          TEST_ASSERT(rund::task::join(handle));
        }

        admitted.store(0u, std::memory_order_relaxed);
        resumed.store(0u, std::memory_order_relaxed);
        saturation_seen = false;
        waiter_results.fill(rund::task::Status{});
        waiter_terminals.fill(rund::task::Status{});
        wake_order.fill(std::numeric_limits<std::uint32_t>::max());

        runtime_task_allocation::Start();
        const rund::task::Handle target = rund::task::spawn(
            "join-wait-target",
            HoldSharedJoinTarget(&admitted, &resumed, kJoinWaiters,
                                 &saturation_seen));
        saturation_handles_valid = static_cast<bool>(target);
        for (std::uint32_t index = 0u; index < kJoinWaiters; ++index) {
          waiter_handles[index] = rund::task::spawn(
              "join-waiter",
              AwaitSharedJoinTarget(target, index, &admitted, &resumed,
                                    waiter_results, wake_order));
          saturation_handles_valid =
              saturation_handles_valid && waiter_handles[index];
        }
        target_terminal = rund::task::join(target);
        for (std::uint32_t index = 0u; index < kJoinWaiters; ++index) {
          waiter_terminals[index] = rund::task::join(waiter_handles[index]);
        }
        runtime_task_allocation::Stop();
        join_allocations = runtime_task_allocation::Count();
      });

  TEST_ASSERT(saturation_report.ok());
  TEST_ASSERT(saturation_handles_valid);
  TEST_ASSERT(saturation_seen);
  TEST_ASSERT(admitted.load(std::memory_order_relaxed) == kJoinWaiters);
  TEST_ASSERT(resumed.load(std::memory_order_relaxed) == kJoinWaiters);
  TEST_ASSERT(target_terminal.ok());
  TEST_ASSERT(join_allocations == 0u);
  for (std::uint32_t index = 0u; index < kJoinWaiters; ++index) {
    TEST_ASSERT(waiter_results[index].ok());
    TEST_ASSERT(waiter_terminals[index].ok());
    TEST_ASSERT(wake_order[index] == index);
  }
  return 0;
}
