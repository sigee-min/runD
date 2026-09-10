#include "local.hpp"

#include "test/assert.hpp"

#include <rund/session.hpp>
#include <rund/task/api.hpp>

#include <array>
#include <atomic>
#include <cstdint>
#include <type_traits>

#include "../../../../../../src/runtime/task/scheduler/progress/ready/pick.hpp"
#include "../../../../../../src/runtime/task/scheduler/state.hpp"
#include "../../../../../../src/runtime/task/scheduler/state/model/work.hpp"
#include "../../../../../../src/runtime/task/scheduler/state/storage/ready/queue.hpp"

namespace ready_queue_detail {
namespace {

struct LaneGate final {
  std::atomic<bool> started{false};
  std::atomic<bool> release{false};
};

void BlockLane(void *const context) noexcept {
  auto &gate = *static_cast<LaneGate *>(context);
  gate.started.store(true, std::memory_order_release);
  gate.started.notify_one();
  while (!gate.release.load(std::memory_order_acquire)) {
    gate.release.wait(false, std::memory_order_acquire);
  }
}

struct LeafOrder final {
  std::uint64_t trace = 0u;
  bool second_calculated_first = false;
};

LeafOrder RunLeafOrder(const bool blocked) {
  LaneGate gate{};
  rund::node::SchedulerWork work{
      .context = &gate,
      .invoke = BlockLane,
  };
  std::atomic<bool> first_calculated{false};
  std::atomic<bool> second_calculated_first{false};
  rund::task::Status joined{};
  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .id = 91u,
          .workers = 1u,
          .scheduler = {
              .task_workers = 2u,
              .task_capacity = 2u,
              .ready_queue_capacity = 2u,
          },
      },
      [&] {
        if (blocked) {
          rund::node::Scheduler *const scheduler =
              rund::node::Scheduler::Active();
          TEST_ASSERT(scheduler != nullptr);
          TEST_ASSERT(scheduler->EnqueueWork(&work, 0u));
          gate.started.wait(false, std::memory_order_acquire);
        }
        std::array<rund::task::Handle, 2u> handles{
            rund::task::spawn("ordered-first", [&] {
              first_calculated.store(true, std::memory_order_release);
            }),
            rund::task::spawn("ordered-second", [&] {
              second_calculated_first.store(
                  !first_calculated.load(std::memory_order_acquire),
                  std::memory_order_release);
              gate.release.store(true, std::memory_order_release);
              gate.release.notify_one();
            }),
        };
        TEST_ASSERT(handles[0]);
        TEST_ASSERT(handles[1]);
        joined = rund::task::join_all(handles);
      });
  TEST_ASSERT(report);
  TEST_ASSERT(joined);
  TEST_ASSERT(first_calculated.load(std::memory_order_acquire));
  return LeafOrder{
      .trace = report.tasks().trace_hash(),
      .second_calculated_first =
          second_calculated_first.load(std::memory_order_acquire),
  };
}


}  // namespace

int CheckStorageAndOrder() {
  using rund::node::ReadyPick;
  using rund::node::ReadyPickDisposition;

  static_assert(!std::is_aggregate_v<ReadyPick>);
  static_assert(!std::is_default_constructible_v<ReadyPick>);
  static_assert(std::is_trivially_copyable_v<ReadyPick>);
  constexpr ReadyPick no_pick = ReadyPick::none();
  static_assert(no_pick.disposition() == ReadyPickDisposition::None);
  static_assert(no_pick.task_id() == 0u);
  constexpr ReadyPick zero_task = ReadyPick::task(0u);
  static_assert(zero_task.disposition() == ReadyPickDisposition::None);
  static_assert(zero_task.task_id() == 0u);
  constexpr ReadyPick task_pick = ReadyPick::task(17u);
  static_assert(task_pick.disposition() == ReadyPickDisposition::Task);
  static_assert(task_pick.task_id() == 17u);
  constexpr ReadyPick blocked_pick = ReadyPick::blocked();
  static_assert(blocked_pick.disposition() == ReadyPickDisposition::Blocked);
  static_assert(blocked_pick.task_id() == 0u);
  constexpr ReadyPick activity_pick = ReadyPick::activity();
  static_assert(activity_pick.disposition() == ReadyPickDisposition::Activity);
  static_assert(activity_pick.task_id() == 0u);

  ReadyQueue queue{};
  queue.configure(4u);
  TEST_ASSERT(queue.capacity() == 4u);
  TEST_ASSERT(queue.empty());

  TEST_ASSERT(queue.push_back(1u));
  TEST_ASSERT(queue.push_back(2u));
  TEST_ASSERT(queue.push_back(3u));
  TEST_ASSERT(queue.push_back(4u));
  TEST_ASSERT(!queue.push_back(5u));

  std::uint64_t id = 0u;
  TEST_ASSERT(queue.pop_front(id) && id == 1u);
  TEST_ASSERT(queue.pop_front(id) && id == 2u);
  TEST_ASSERT(queue.push_back(5u));
  TEST_ASSERT(queue.push_back(6u));

  TEST_ASSERT(queue.front(id) && id == 3u);

  TEST_ASSERT(queue.remove(4u));
  TEST_ASSERT(queue.push_front(2u));
  TEST_ASSERT(queue.take_first(
      [](const std::uint64_t candidate) { return candidate == 5u; }, id));
  TEST_ASSERT(id == 5u);

  TEST_ASSERT(queue.pop_front(id) && id == 2u);
  TEST_ASSERT(queue.pop_front(id) && id == 3u);
  TEST_ASSERT(queue.pop_front(id) && id == 6u);
  TEST_ASSERT(queue.empty());
  TEST_ASSERT(!queue.front(id));
  TEST_ASSERT(!queue.pop_front(id));

  const LeafOrder ordinary_order = RunLeafOrder(false);
  const LeafOrder blocked_order = RunLeafOrder(true);
  TEST_ASSERT(blocked_order.second_calculated_first);
  TEST_ASSERT(blocked_order.trace == ordinary_order.trace);

  queue.clear();
  TEST_ASSERT(queue.push_front(9u));
  TEST_ASSERT(queue.pop_front(id) && id == 9u);

  return 0;
}

}  // namespace ready_queue_detail
