#include "coroutine/allocation.hpp"
#include "test/assert.hpp"

#include <rund/session.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace {

constexpr std::size_t kWorkers = 4u;
constexpr std::size_t kLeafTasks = 8u;
constexpr std::size_t kTasksPerLeaf = kWorkers;
constexpr std::size_t kTasks = kLeafTasks * kTasksPerLeaf;

rund::task::Task<void> Complete() { co_return; }

} // namespace

int RunRuntimeTaskLaneContract() {
  rund::Session runtime{};
  rund::SessionConfig config{};
  config.id = 83u;
  config.workers = 1u;
  config.scheduler.task_workers = kWorkers;
  config.scheduler.task_capacity = kTasks;
  config.scheduler.ready_queue_capacity = kTasks;
  TEST_ASSERT(runtime.open(config));

  std::atomic<std::uint64_t> completed{0u};
  std::uint64_t allocations = 0u;
  const rund::Session::Result report = runtime.scope([&] {
    const auto run = [&](const bool fail) {
      std::array<rund::task::Handle, kTasks> handles{};
      for (std::size_t leaf = 0u; leaf < kLeafTasks; ++leaf) {
        const std::size_t base = leaf * kTasksPerLeaf;
        handles[base] = rund::task::spawn("lane-leaf", [&, fail] {
          completed.fetch_add(1u, std::memory_order_relaxed);
          if (fail) {
            (void)rund::task::yield();
          }
        });
        for (std::size_t offset = 1u; offset < kTasksPerLeaf; ++offset) {
          handles[base + offset] =
              rund::task::spawn("lane-decoy", Complete());
        }
      }
      for (const rund::task::Handle handle : handles) {
        TEST_ASSERT(handle);
      }
      return rund::task::join_all(handles);
    };

    TEST_ASSERT(run(false));
    runtime_task_allocation::Start();
    const rund::task::Status failure = run(true);
    runtime_task_allocation::Stop();
    allocations = runtime_task_allocation::Count();
    TEST_ASSERT(!failure);
    TEST_ASSERT(failure.code() ==
                rund::ReasonCode::TaskLeafPrimitiveForbidden);
  });

  TEST_ASSERT(report);
  TEST_ASSERT(completed.load(std::memory_order_relaxed) == kLeafTasks * 2u);
  TEST_ASSERT(allocations == 0u);
  TEST_ASSERT(runtime.close());
  return 0;
}
