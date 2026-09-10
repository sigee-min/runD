#include "local.hpp"

#include "test/assert.hpp"

#include <rund/session.hpp>
#include <rund/task/api.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ready_queue_detail {

int CheckReuseAndBatch() {
  constexpr std::size_t kRounds = 64u;
  constexpr std::size_t kTasks = 4u;
  std::uint64_t completed = 0u;
  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
        .workers = 1u,
        .scheduler = {
          .task_workers = 1u,
          .task_capacity = 4u,
          .ready_queue_capacity = 4u,
        },
      },
      [&] {
        std::array<rund::task::Handle, kTasks> handles{};
        for (std::size_t round = 0u; round < kRounds; ++round) {
          for (auto& handle : handles) {
            handle = rund::task::spawn("index-reuse", [&completed] {
              ++completed;
            });
            TEST_ASSERT(handle);
          }
          TEST_ASSERT(rund::task::join_all(handles));
        }
      });
  TEST_ASSERT(report);
  TEST_ASSERT(completed == kRounds * kTasks);

  completed = 0u;
  const rund::Session::Result wrap_report = rund::run(
      rund::SessionConfig{
        .workers = 1u,
        .scheduler = {
          .task_workers = 1u,
          .task_capacity = 4u,
          .ready_queue_capacity = 4u,
          .channel_capacity = 1u,
          .channel_wait_capacity = 4u,
        },
      },
      [&] {
        auto gate = rund::task::channel<std::uint32_t>::make(0u);
        const auto held = rund::task::spawn("index-held", HoldIndex(&gate));
        TEST_ASSERT(held);
        std::array<rund::task::Handle, 3u> handles{};
        for (std::size_t round = 0u; round < kRounds; ++round) {
          for (auto& handle : handles) {
            handle = rund::task::spawn("index-wrap", [&completed] {
              ++completed;
            });
            TEST_ASSERT(handle);
          }
          TEST_ASSERT(rund::task::join_all(handles));
        }
        TEST_ASSERT(gate.close());
        TEST_ASSERT(rund::task::join(held));
      });
  TEST_ASSERT(wrap_report);
  TEST_ASSERT(completed == kRounds * 3u);

  constexpr std::size_t kBatchTasks = 4096u;
  std::atomic<std::uint64_t> batch_completed{0u};
  const rund::Session::Result batch_report = rund::run(
      rund::SessionConfig{
        .workers = 1u,
        .scheduler = {
          .task_workers = 4u,
          .task_capacity = kBatchTasks,
          .ready_queue_capacity = kBatchTasks,
        },
      },
      [&] {
        std::vector<rund::task::Handle> handles{};
        handles.reserve(kBatchTasks);
        for (std::size_t index = 0u; index < kBatchTasks; ++index) {
          handles.push_back(rund::task::spawn("index-batch", [&] {
            batch_completed.fetch_add(1u, std::memory_order_relaxed);
          }));
          TEST_ASSERT(handles.back());
        }
        TEST_ASSERT(rund::task::join_all(handles));
      });
  TEST_ASSERT(batch_report);
  TEST_ASSERT(batch_completed.load(std::memory_order_relaxed) == kBatchTasks);
  TEST_ASSERT(batch_report.tasks().spawned() == kBatchTasks);
  TEST_ASSERT(batch_report.tasks().completed() == kBatchTasks);
  TEST_ASSERT(batch_report.tasks().failed() == 0u);
  TEST_ASSERT(batch_report.tasks().callable_resets() == kBatchTasks);
  TEST_ASSERT(batch_report.tasks().participating_task_workers() == 4u);
  TEST_ASSERT(batch_report.tasks().lane_dispatch_batch_packets() == 1u);
  TEST_ASSERT(batch_report.tasks().lane_dispatch_batch_logical_tasks() ==
              kBatchTasks);
  TEST_ASSERT(batch_report.tasks().lane_dispatch_batch_scratch_reuses() >= 1u);

  std::atomic<std::uint64_t> mixed_completed{0u};
  const rund::Session::Result mixed_report = rund::run(
      rund::SessionConfig{
        .workers = 1u,
        .scheduler = {
          .task_workers = 4u,
          .task_capacity = 8u,
          .ready_queue_capacity = 8u,
        },
      },
      [&] {
        std::array<rund::task::Handle, 8u> handles{};
        for (std::size_t index = 0u; index < handles.size(); ++index) {
          if ((index & 1u) == 0u) {
            handles[index] = rund::task::spawn("mixed-leaf", [&] {
              mixed_completed.fetch_add(1u, std::memory_order_relaxed);
            });
          } else {
            handles[index] = rund::task::spawn(
                "mixed-coroutine", CompleteIndex(&mixed_completed));
          }
          TEST_ASSERT(handles[index]);
        }
        TEST_ASSERT(rund::task::join_all(handles));
      });
  TEST_ASSERT(mixed_report);
  TEST_ASSERT(mixed_completed.load(std::memory_order_relaxed) == 8u);
  TEST_ASSERT(mixed_report.tasks().completed() == 8u);
  TEST_ASSERT(mixed_report.tasks().failed() == 0u);

  return 0;
}

}  // namespace ready_queue_detail
