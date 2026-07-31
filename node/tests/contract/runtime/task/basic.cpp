#include "coroutine/allocation.hpp"
#include "test/assert.hpp"

#include <rund/session.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <mutex>
#include <string_view>
#include <vector>

namespace {

struct ScopeAllocationProbe final {
  std::uint32_t *calls = nullptr;

  explicit ScopeAllocationProbe(std::uint32_t &count) noexcept
      : calls(&count) {}
  ScopeAllocationProbe(const ScopeAllocationProbe &) = delete;
  ScopeAllocationProbe &operator=(const ScopeAllocationProbe &) = delete;
  ScopeAllocationProbe(ScopeAllocationProbe &&) = delete;
  ScopeAllocationProbe &operator=(ScopeAllocationProbe &&) = delete;

  void operator()() const noexcept { ++*calls; }
};

} // namespace

int RunRuntimeTaskBasicContract() {
  rund::Session retry_runtime{};
  rund::SessionConfig retry_options{};
  retry_options.id = 78u;
  retry_options.workers = 1u;
  runtime_task_allocation::FailNext();
  const rund::Session::Status allocation_failure =
      retry_runtime.open(retry_options);
  TEST_ASSERT(!allocation_failure);
  TEST_ASSERT(allocation_failure.error() ==
              std::string_view{"task_scheduler_allocation_failed"});
  TEST_ASSERT(retry_runtime.snapshot().state ==
              rund::SessionState::Unconfigured);
  TEST_ASSERT(!retry_runtime.resources());
  TEST_ASSERT(retry_runtime.open(retry_options));
  TEST_ASSERT(retry_runtime.close());

  rund::Session late_retry_runtime{};
  rund::SessionConfig late_retry_options = retry_options;
  late_retry_options.id = 79u;
  late_retry_options.scheduler.coroutine_frame_alignment = 3u;
  const rund::Session::Status late_failure =
      late_retry_runtime.open(late_retry_options);
  TEST_ASSERT(!late_failure);
  TEST_ASSERT(late_failure.code() ==
              rund::ReasonCode::TaskFrameLimitsInvalid);
  TEST_ASSERT(late_retry_runtime.snapshot().state ==
              rund::SessionState::Unconfigured);
  TEST_ASSERT(!late_retry_runtime.resources());
  late_retry_options.scheduler.coroutine_frame_alignment = 16u;
  TEST_ASSERT(late_retry_runtime.open(late_retry_options));
  TEST_ASSERT(late_retry_runtime.close());

  const rund::Session::Result terminal_capture = rund::run(
      rund::SessionConfig{
          .id = 80u,
          .workers = 1u,
          .trace_capacity = 8u,
          .scheduler =
              {
                  .task_workers = 1u,
                  .task_capacity = 2u,
                  .ready_queue_capacity = 2u,
                  .observation_capacity = 8u,
                  .host_event_capacity = 8u,
              },
      },
      [] {
        TEST_ASSERT(
            rund::task::join(rund::task::spawn("terminal-capture", [] {})));
        runtime_task_allocation::Start();
      });
  runtime_task_allocation::Stop();
  const std::uint64_t terminal_capture_allocations =
      runtime_task_allocation::Count();
  TEST_ASSERT(terminal_capture);
  TEST_ASSERT(!terminal_capture.trace().records.empty());
  TEST_ASSERT(terminal_capture_allocations == 0u);
  TEST_ASSERT(terminal_capture.tasks().callable_inline_stores() == 1u);
  TEST_ASSERT(terminal_capture.tasks().callable_resets() == 1u);
  TEST_ASSERT(terminal_capture.tasks().root_join_ready_fast_paths() == 1u);

  const rund::task::Handle outside_task = rund::task::spawn("outside", [] {});
  TEST_ASSERT(!outside_task);
  TEST_ASSERT(outside_task.error() == std::string_view{"node_runtime_missing"});
  const rund::task::YieldOp outside_yield = rund::task::yield();
  TEST_ASSERT(!outside_yield);
  TEST_ASSERT(outside_yield.error() ==
              std::string_view{"node_runtime_missing"});

  rund::task::YieldOp runtime_non_task_yield = rund::task::yield();
  const rund::Session::Result non_task_yield_report = rund::run(
      rund::SessionConfig{
          .workers = 2u,
          .scheduler =
              {
                  .task_capacity = 2u,
                  .ready_queue_capacity = 2u,
              },
      },
      [&] { runtime_non_task_yield = rund::task::yield(); });
  TEST_ASSERT(non_task_yield_report.ok());
  TEST_ASSERT(!runtime_non_task_yield);
  TEST_ASSERT(runtime_non_task_yield.error() ==
              std::string_view{"task_context_missing"});

  rund::Session *scope_runtime = nullptr;
  rund::task::Status scope_join{};
  const rund::Session::Result scope_report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_capacity = 2u,
                  .ready_queue_capacity = 2u,
              },
      },
      [&](rund::Session &runtime) {
        scope_runtime = &runtime;
        scope_join =
            rund::task::join(rund::task::spawn("runtime-handle", [] {}));
      });
  TEST_ASSERT(scope_report.ok());
  TEST_ASSERT(scope_runtime != nullptr);
  TEST_ASSERT(scope_join.ok());

  rund::Session warm_runtime{};
  rund::SessionConfig warm_options{};
  warm_options.id = 79u;
  warm_options.workers = 1u;
  warm_options.scheduler.task_capacity = 2u;
  warm_options.scheduler.ready_queue_capacity = 2u;
  warm_options.scheduler.task_workers = 1u;
  TEST_ASSERT(warm_runtime.open(warm_options));
  std::uint64_t warm_submit_allocations = 0u;
  rund::task::Status warm_submit_join{};
  const rund::Session::Result warm_submit_scope = warm_runtime.scope([&] {
    TEST_ASSERT(rund::task::join(rund::task::spawn("submit-prime", [] {})));
    runtime_task_allocation::Start();
    const rund::task::Handle task = rund::task::spawn("submit-warm", [] {});
    warm_submit_join = rund::task::join(task);
    runtime_task_allocation::Stop();
    warm_submit_allocations = runtime_task_allocation::Count();
  });
  TEST_ASSERT(warm_submit_scope);
  TEST_ASSERT(warm_submit_join);
  TEST_ASSERT(warm_submit_allocations == 0u);

  std::uint32_t scope_callback_calls = 0u;
  ScopeAllocationProbe scope_callback{scope_callback_calls};
  TEST_ASSERT(warm_runtime.scope(scope_callback));
  runtime_task_allocation::Start();
  const rund::Session::Result allocation_free_scope =
      warm_runtime.scope(scope_callback);
  runtime_task_allocation::Stop();
  TEST_ASSERT(allocation_free_scope);
  TEST_ASSERT(scope_callback_calls == 2u);
  TEST_ASSERT(runtime_task_allocation::Count() == 0u);
  TEST_ASSERT(warm_runtime.close());

  constexpr std::size_t segment_tasks = 64u;
  rund::Session segment_runtime{};
  rund::SessionConfig segment_options{};
  segment_options.id = 80u;
  segment_options.workers = 1u;
  segment_options.scheduler.task_capacity = segment_tasks;
  segment_options.scheduler.ready_queue_capacity = segment_tasks;
  segment_options.scheduler.task_workers = 4u;
  TEST_ASSERT(segment_runtime.open(segment_options));
  std::array<rund::task::Handle, segment_tasks> segment_handles{};
  std::uint64_t warm_segment_allocations = 0u;
  const rund::Session::Result segment_scope = segment_runtime.scope([&] {
    const auto run_batch = [&] {
      for (rund::task::Handle &handle : segment_handles) {
        handle = rund::task::spawn("segment-warm", [] {});
      }
      return rund::task::join_all(segment_handles);
    };
    TEST_ASSERT(run_batch());
    runtime_task_allocation::Start();
    const rund::task::Status joined = run_batch();
    runtime_task_allocation::Stop();
    warm_segment_allocations = runtime_task_allocation::Count();
    TEST_ASSERT(joined);
  });
  TEST_ASSERT(segment_scope);
  TEST_ASSERT(warm_segment_allocations == 0u);
  TEST_ASSERT(segment_scope.tasks().participating_task_workers() == 4u);
  TEST_ASSERT(segment_runtime.close());

  std::vector<int> task_order{};
  std::mutex task_order_mutex{};
  rund::task::Status task_join{};
  const rund::Session::Result task_report = rund::run(
      rund::SessionConfig{
          .id = 78u,
          .workers = 2u,
          .scheduler =
              {
                  .task_capacity = 8u,
                  .ready_queue_capacity = 8u,
              },
      },
      [&] {
        const rund::task::Handle first = rund::task::spawn("first", [&] {
          std::lock_guard<std::mutex> lock(task_order_mutex);
          task_order.push_back(1);
        });
        const rund::task::Handle second = rund::task::spawn("second", [&] {
          std::lock_guard<std::mutex> lock(task_order_mutex);
          task_order.push_back(2);
        });
        const rund::task::Handle third = rund::task::spawn("third", [&] {
          std::lock_guard<std::mutex> lock(task_order_mutex);
          task_order.push_back(3);
        });
        task_join = rund::task::join(first, second, third);
      });
  TEST_ASSERT(task_report.ok());
  TEST_ASSERT(task_join.ok());
  TEST_ASSERT(task_report.tasks().spawned() == 3u);
  TEST_ASSERT(task_report.tasks().completed() == 3u);
  TEST_ASSERT(task_report.tasks().yields() == 0u);
  TEST_ASSERT(task_report.tasks().spawn_task_id_range_reservations() == 1u);
  TEST_ASSERT(task_report.tasks().spawn_task_id_range_used_slots() == 3u);
  TEST_ASSERT(task_report.tasks().spawn_per_task_id_allocations_avoided() ==
              2u);
  TEST_ASSERT(task_report.tasks().task_workers() == 2u);
  TEST_ASSERT(task_report.tasks().participating_task_workers() == 2u);
  TEST_ASSERT(task_order.size() == 3u);
  std::sort(task_order.begin(), task_order.end());
  for (std::size_t index = 0u; index < task_order.size(); ++index) {
    TEST_ASSERT(task_order[index] == static_cast<int>(index + 1u));
  }

  const auto named_trace = [](const char *const name) {
    const rund::Session::Result report = rund::run(
        rund::SessionConfig{
            .id = 81u,
            .workers = 1u,
            .scheduler =
                {
                    .task_capacity = 1u,
                    .ready_queue_capacity = 1u,
                },
        },
        [&] { TEST_ASSERT(rund::task::join(rund::task::spawn(name, [] {}))); });
    TEST_ASSERT(report.ok());
    return report.tasks().trace_hash();
  };
  const std::uint64_t first_named_trace = named_trace("named-alpha");
  TEST_ASSERT(named_trace("named-alpha") == first_named_trace);
  TEST_ASSERT(named_trace("named-beta") != first_named_trace);

  rund::task::Handle empty_name{};
  rund::task::Handle null_name{};
  const rund::Session::Result invalid_name_report = rund::run(
      rund::SessionConfig{
          .id = 82u,
          .workers = 1u,
          .scheduler =
              {
                  .task_capacity = 1u,
                  .ready_queue_capacity = 1u,
              },
      },
      [&] {
        empty_name = rund::task::spawn("", [] {});
        const char *const name = nullptr;
        null_name = rund::task::spawn(name, [] {});
      });
  TEST_ASSERT(invalid_name_report.ok());
  TEST_ASSERT(!empty_name);
  TEST_ASSERT(empty_name.code() == rund::ReasonCode::TaskInvalid);
  TEST_ASSERT(!null_name);
  TEST_ASSERT(null_name.code() == rund::ReasonCode::TaskInvalid);
  return 0;
}
