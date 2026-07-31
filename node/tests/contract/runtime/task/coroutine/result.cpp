#include "allocation.hpp"
#include "src/runtime/task/scheduler/task/completion/value.hpp"
#include "test/assert.hpp"

#include <rund/session.hpp>
#include <rund/task/api.hpp>
#include <rund/task/cancel.hpp>
#include <rund/task/handle/typed.hpp>
#include <rund/task/result.hpp>

#include <array>
#include <atomic>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

static_assert(std::is_default_constructible_v<rund::task::Status>);
static_assert(std::is_trivially_copyable_v<rund::task::Status>);
static_assert(std::is_trivially_destructible_v<rund::task::Status>);
static_assert(sizeof(rund::task::Status) == 2u);
static_assert(alignof(rund::task::Status) == 2u);

static_assert(std::is_trivially_copyable_v<rund::task::Handle>);
static_assert(std::is_trivially_destructible_v<rund::task::Handle>);
static_assert(sizeof(rund::task::Handle) == 32u);
static_assert(alignof(rund::task::Handle) == 8u);

static_assert(std::is_same_v<
              decltype(rund::task::join(std::declval<rund::task::Handle>())),
              rund::task::Status>);
static_assert(std::is_same_v<
              decltype(std::declval<rund::task::stop_source>().request_stop()),
              rund::task::Status>);
static_assert(
    std::is_same_v<decltype(std::declval<rund::task::stop_token>().state()),
                   rund::task::StopState>);

namespace {

struct ThrowingMove final {
  explicit ThrowingMove(const int initial) noexcept : value(initial) {}
  ThrowingMove(const ThrowingMove &) = delete;
  ThrowingMove &operator=(const ThrowingMove &) = delete;
  ThrowingMove(ThrowingMove &&) { throw 1; }
  ThrowingMove &operator=(ThrowingMove &&) { throw 1; }

  int value = 0;
};

struct UnevenMove final {
  explicit UnevenMove(const int initial) noexcept : value(initial) {}
  UnevenMove(const UnevenMove &) = delete;
  UnevenMove &operator=(const UnevenMove &) = delete;
  UnevenMove(UnevenMove &&) { throw 2; }
  UnevenMove &operator=(UnevenMove &&other) noexcept {
    value = other.value;
    return *this;
  }

  int value = 0;
};

static_assert(
    !std::is_nothrow_move_assignable_v<rund::task::Result<UnevenMove>>);

void CountWake(void *const value) noexcept {
  static_cast<std::atomic<std::uint32_t> *>(value)->fetch_add(
      1u, std::memory_order_relaxed);
}

struct WakeOrder final {
  std::array<std::uint32_t, 3u> values{};
  std::uint32_t count{};
};

struct OrderedWake final {
  WakeOrder *order{};
  std::uint32_t value{};
};

void RecordWake(void *const value) noexcept {
  auto &wake = *static_cast<OrderedWake *>(value);
  wake.order->values[wake.order->count] = wake.value;
  ++wake.order->count;
}

} // namespace

int RunRuntimeTaskResultContract() {
  const rund::task::Status not_started{};
  TEST_ASSERT(!not_started);
  TEST_ASSERT(not_started.code() == rund::ReasonCode::TaskInvalid);
  TEST_ASSERT(not_started.error() == "task_invalid");
  TEST_ASSERT(not_started.exit_code() == 1);

  const rund::task::Status status =
      rund::task::Status::fail(rund::ReasonCode::TaskFailed);
  TEST_ASSERT(!status);
  TEST_ASSERT(status.code() == rund::ReasonCode::TaskFailed);
  TEST_ASSERT(status.error() == "task_failed");
  TEST_ASSERT(status.exit_code() == 1);

  rund::task::Result<int> ready = rund::task::Result<int>::success(7);
  TEST_ASSERT(ready);
  TEST_ASSERT(ready.ok());
  TEST_ASSERT(ready.operator->() != nullptr);
  TEST_ASSERT(*ready == 7);
  TEST_ASSERT(ready.code() == rund::ReasonCode::Ok);
  TEST_ASSERT(ready.error().empty());
  TEST_ASSERT(ready.exit_code() == 0);

  const rund::task::Result<int> failed =
      rund::task::Result<int>::fail(rund::ReasonCode::TaskCancelled);
  TEST_ASSERT(!failed);
  TEST_ASSERT(!failed.ok());
  TEST_ASSERT(failed.operator->() == nullptr);
  TEST_ASSERT(failed.code() == rund::ReasonCode::TaskCancelled);
  TEST_ASSERT(failed.error() == "task_cancelled");
  TEST_ASSERT(failed.exit_code() == 1);

  const rund::task::Result<rund::task::Status> status_value =
      rund::task::Result<rund::task::Status>::success(
          rund::task::Status::success());
  TEST_ASSERT(status_value);
  TEST_ASSERT(status_value->ok());
  TEST_ASSERT(status_value.code() == rund::ReasonCode::Ok);

  rund::task::Result<ThrowingMove> invalidated =
      rund::task::Result<ThrowingMove>::fail(rund::ReasonCode::TaskCancelled);
  rund::task::Result<ThrowingMove> moving =
      rund::task::Result<ThrowingMove>::success(7);
  try {
    invalidated = std::move(moving);
    TEST_ASSERT(false);
  } catch (const int) {
  }
  TEST_ASSERT(!invalidated);
  TEST_ASSERT(invalidated.operator->() == nullptr);
  TEST_ASSERT(invalidated.code() == rund::ReasonCode::TaskFailed);
  TEST_ASSERT(invalidated.error() == "task_failed");
  TEST_ASSERT(invalidated.exit_code() == 1);

  rund::task::Result<UnevenMove> uneven =
      rund::task::Result<UnevenMove>::fail(rund::ReasonCode::TaskCancelled);
  rund::task::Result<UnevenMove> uneven_value =
      rund::task::Result<UnevenMove>::success(7);
  try {
    uneven = std::move(uneven_value);
    TEST_ASSERT(false);
  } catch (const int) {
  }
  TEST_ASSERT(!uneven);
  TEST_ASSERT(uneven.operator->() == nullptr);
  TEST_ASSERT(uneven.code() == rund::ReasonCode::TaskFailed);
  TEST_ASSERT(uneven.error() == "task_failed");
  TEST_ASSERT(uneven.exit_code() == 1);

  const rund::task::Result<void> complete = rund::task::Result<void>::success();
  TEST_ASSERT(complete);
  TEST_ASSERT(complete.ok());
  TEST_ASSERT(complete.code() == rund::ReasonCode::Ok);
  TEST_ASSERT(complete.exit_code() == 0);

  rund::node::CompletionPool pool{};
  TEST_ASSERT(pool.configure(rund::node::CompletionLimits{
      .capacity = 1u, .result_bytes = 64u, .result_alignment = 64u}));
  TEST_ASSERT(pool.resident_cells() == 0u);
  const rund::node::CompletionLease first = pool.claim();
  TEST_ASSERT(first);
  TEST_ASSERT(pool.resident_cells() == 1u);
  TEST_ASSERT(!pool.claim());
  TEST_ASSERT(!rund::node::CompletionPool::transition(
      first, rund::task::Phase::Running));
  TEST_ASSERT(rund::node::CompletionPool::poll(first).phase ==
              rund::task::Phase::Admitted);
  TEST_ASSERT(
      rund::node::CompletionPool::transition(first, rund::task::Phase::Ready));
  TEST_ASSERT(rund::node::CompletionPool::transition(
      first, rund::task::Phase::Running));
  TEST_ASSERT(rund::node::CompletionPool::transition(
      first, rund::task::Phase::Committing));
  TEST_ASSERT(rund::node::CompletionPool::complete(first, 7));
  TEST_ASSERT(rund::node::CompletionPool::wait(first));
  const rund::task::Result<int> stored =
      rund::node::CompletionPool::read<int>(first);
  TEST_ASSERT(stored && *stored == 7);
  TEST_ASSERT(!rund::node::CompletionPool::read<std::uint32_t>(first));
  rund::node::CompletionPool::release(first);
  TEST_ASSERT(rund::node::CompletionPool::poll(first).code ==
              rund::ReasonCode::TaskHandleStale);

  rund::node::CompletionPool paged{};
  TEST_ASSERT(paged.configure(rund::node::CompletionLimits{
      .capacity = 257u, .result_bytes = 64u, .result_alignment = 64u}));
  const rund::node::CompletionLease anchor = paged.claim();
  rund::detail::task::ResultHandle<void> anchor_handle =
      paged.observe<void>(anchor);
  std::vector<rund::node::CompletionLease> first_page{};
  first_page.reserve(255u);
  for (std::uint32_t index = 0u; index < 255u; ++index) {
    first_page.push_back(paged.claim());
    TEST_ASSERT(first_page.back());
  }
  TEST_ASSERT(paged.resident_cells() == 256u);
  std::atomic_bool boundary_claimed{false};
  rund::node::CompletionLease boundary{};
  std::thread grow_page{[&] {
    boundary = paged.claim();
    boundary_claimed.store(true, std::memory_order_release);
  }};
  while (!boundary_claimed.load(std::memory_order_acquire)) {
    TEST_ASSERT(anchor_handle.poll().phase == rund::task::Phase::Admitted);
  }
  grow_page.join();
  TEST_ASSERT(boundary);
  TEST_ASSERT(paged.resident_cells() == 257u);
  const auto finish = [](const rund::node::CompletionLease lease) {
    TEST_ASSERT(rund::node::CompletionPool::transition(
        lease, rund::task::Phase::Ready));
    TEST_ASSERT(rund::node::CompletionPool::transition(
        lease, rund::task::Phase::Running));
    TEST_ASSERT(rund::node::CompletionPool::transition(
        lease, rund::task::Phase::Committing));
    TEST_ASSERT(rund::node::CompletionPool::complete(lease));
    rund::node::CompletionPool::release(lease);
  };
  finish(anchor);
  anchor_handle = {};
  for (const rund::node::CompletionLease lease : first_page) {
    finish(lease);
  }
  finish(boundary);

  runtime_task_allocation::Start();
  const rund::node::CompletionLease warm = pool.claim();
  const bool warm_ready =
      warm &&
      rund::node::CompletionPool::transition(warm, rund::task::Phase::Ready) &&
      rund::node::CompletionPool::transition(warm,
                                             rund::task::Phase::Running) &&
      rund::node::CompletionPool::transition(warm,
                                             rund::task::Phase::Committing) &&
      rund::node::CompletionPool::complete(warm, 9);
  runtime_task_allocation::Stop();
  TEST_ASSERT(warm_ready);
  TEST_ASSERT(runtime_task_allocation::Count() == 0u);
  rund::node::CompletionPool::release(warm);

  rund::node::CompletionLease escaped{};
  {
    rund::node::CompletionPool owner{};
    TEST_ASSERT(owner.configure(rund::node::CompletionLimits{
        .capacity = 1u, .result_bytes = 64u, .result_alignment = 64u}));
    escaped = owner.claim();
    TEST_ASSERT(rund::node::CompletionPool::transition(
        escaped, rund::task::Phase::Ready));
    TEST_ASSERT(rund::node::CompletionPool::transition(
        escaped, rund::task::Phase::Running));
    TEST_ASSERT(rund::node::CompletionPool::transition(
        escaped, rund::task::Phase::Committing));
    TEST_ASSERT(rund::node::CompletionPool::complete(escaped, 11));
  }
  const rund::task::Result<int> escaped_result =
      rund::node::CompletionPool::read<int>(escaped);
  TEST_ASSERT(escaped_result && *escaped_result == 11);
  rund::node::CompletionPool::release(escaped);

  rund::node::CompletionPool handles{};
  TEST_ASSERT(handles.configure(rund::node::CompletionLimits{
      .capacity = 1u, .result_bytes = 64u, .result_alignment = 64u}));
  const rund::node::CompletionLease producer = handles.claim();
  rund::detail::task::ResultHandle<int> handle = handles.observe<int>(producer);
  TEST_ASSERT(handle);
  TEST_ASSERT(handle.poll().phase == rund::task::Phase::Admitted);
  TEST_ASSERT(rund::node::CompletionPool::transition(producer,
                                                     rund::task::Phase::Ready));
  TEST_ASSERT(rund::node::CompletionPool::transition(
      producer, rund::task::Phase::Running));
  TEST_ASSERT(rund::node::CompletionPool::transition(
      producer, rund::task::Phase::Committing));
  TEST_ASSERT(rund::node::CompletionPool::complete(producer, 13));
  rund::node::CompletionPool::release(producer);
  TEST_ASSERT(handle.wait());
  const rund::task::Result<int> handled = handle.result();
  TEST_ASSERT(handled && *handled == 13);
  TEST_ASSERT(!handles.claim());
  rund::task::Status worker_wait = rund::task::Status::success();
  rund::task::Status worker_join{};
  const rund::Session::Result worker_report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_capacity = 2u,
                  .ready_queue_capacity = 2u,
              },
      },
      [&] {
        const rund::task::Handle waiter = rund::task::spawn(
            "completion-worker-wait", [&] { worker_wait = handle.wait(); });
        worker_join = rund::task::join(waiter);
      });
  TEST_ASSERT(worker_report.ok());
  TEST_ASSERT(worker_join.ok());
  TEST_ASSERT(!worker_wait);
  TEST_ASSERT(worker_wait.code() == rund::ReasonCode::TaskWorkerWaitForbidden);
  handle = {};
  TEST_ASSERT(handles.claim());

  rund::node::CompletionPool void_handles{};
  TEST_ASSERT(void_handles.configure(rund::node::CompletionLimits{
      .capacity = 1u, .result_bytes = 64u, .result_alignment = 64u}));
  const rund::node::CompletionLease void_producer = void_handles.claim();
  rund::detail::task::ResultHandle<void> void_handle =
      void_handles.observe<void>(void_producer);
  TEST_ASSERT(void_handle);
  TEST_ASSERT(rund::node::CompletionPool::transition(void_producer,
                                                     rund::task::Phase::Ready));
  TEST_ASSERT(rund::node::CompletionPool::transition(
      void_producer, rund::task::Phase::Running));
  TEST_ASSERT(rund::node::CompletionPool::transition(
      void_producer, rund::task::Phase::Committing));
  TEST_ASSERT(rund::node::CompletionPool::complete(void_producer));
  rund::node::CompletionPool::release(void_producer);
  TEST_ASSERT(void_handle.wait());
  TEST_ASSERT(void_handle.result());
  void_handle = {};
  TEST_ASSERT(void_handles.claim());

  rund::node::CompletionPool waiters{};
  TEST_ASSERT(waiters.configure(rund::node::CompletionLimits{
      .capacity = 1u, .result_bytes = 64u, .result_alignment = 64u}));
  const rund::node::CompletionLease waited = waiters.claim();
  std::atomic<std::uint32_t> wake_count{0u};
  rund::node::CompletionWaiter waiter_a{.wake = &CountWake,
                                        .value = &wake_count};
  rund::node::CompletionWaiter waiter_b{.wake = &CountWake,
                                        .value = &wake_count};
  TEST_ASSERT(rund::node::CompletionPool::park(waited, waiter_a));
  TEST_ASSERT(rund::node::CompletionPool::park(waited, waiter_b));
  TEST_ASSERT(rund::node::CompletionPool::unpark(waited, waiter_b));
  TEST_ASSERT(!rund::node::CompletionPool::unpark(waited, waiter_b));
  TEST_ASSERT(
      rund::node::CompletionPool::transition(waited, rund::task::Phase::Ready));
  TEST_ASSERT(rund::node::CompletionPool::transition(
      waited, rund::task::Phase::Running));
  TEST_ASSERT(rund::node::CompletionPool::transition(
      waited, rund::task::Phase::Committing));
  runtime_task_allocation::Start();
  const rund::task::Status waiter_complete =
      rund::node::CompletionPool::complete(waited);
  runtime_task_allocation::Stop();
  TEST_ASSERT(waiter_complete);
  TEST_ASSERT(runtime_task_allocation::Count() == 0u);
  TEST_ASSERT(wake_count.load(std::memory_order_relaxed) == 1u);
  TEST_ASSERT(!rund::node::CompletionPool::park(waited, waiter_b));
  TEST_ASSERT(wake_count.load(std::memory_order_relaxed) == 1u);
  rund::node::CompletionPool::release(waited);

  const rund::node::CompletionLease rejected = waiters.claim();
  TEST_ASSERT(rejected);
  std::atomic<std::uint32_t> reject_wakes{0u};
  rund::node::CompletionWaiter reject_waiter{.wake = &CountWake,
                                             .value = &reject_wakes};
  TEST_ASSERT(rund::node::CompletionPool::park(rejected, reject_waiter));
  runtime_task_allocation::Start();
  const rund::task::Status rejected_once =
      rund::node::CompletionPool::terminate(rejected,
                                            rund::ReasonCode::TaskCancelled);
  runtime_task_allocation::Stop();
  TEST_ASSERT(rejected_once);
  TEST_ASSERT(runtime_task_allocation::Count() == 0u);
  TEST_ASSERT(reject_wakes.load(std::memory_order_relaxed) == 1u);
  const rund::task::Poll rejected_poll =
      rund::node::CompletionPool::poll(rejected);
  TEST_ASSERT(rejected_poll.phase == rund::task::Phase::Cancelled);
  TEST_ASSERT(rejected_poll.code == rund::ReasonCode::TaskCancelled);
  TEST_ASSERT(!rund::node::CompletionPool::terminate(
      rejected, rund::ReasonCode::TaskFailed));
  TEST_ASSERT(rund::node::CompletionPool::poll(rejected).code ==
              rund::ReasonCode::TaskCancelled);
  rund::node::CompletionPool::release(rejected);
  TEST_ASSERT(rund::node::CompletionPool::poll(rejected).code ==
              rund::ReasonCode::TaskHandleStale);

  const rund::node::CompletionLease ordered = waiters.claim();
  TEST_ASSERT(ordered);
  WakeOrder order{};
  OrderedWake ordered_a{.order = &order, .value = 1u};
  OrderedWake ordered_b{.order = &order, .value = 2u};
  OrderedWake ordered_c{.order = &order, .value = 3u};
  rund::node::CompletionWaiter order_a{.wake = &RecordWake,
                                       .value = &ordered_a};
  rund::node::CompletionWaiter order_b{.wake = &RecordWake,
                                       .value = &ordered_b};
  rund::node::CompletionWaiter order_c{.wake = &RecordWake,
                                       .value = &ordered_c};
  TEST_ASSERT(rund::node::CompletionPool::park(ordered, order_a));
  TEST_ASSERT(rund::node::CompletionPool::park(ordered, order_b));
  TEST_ASSERT(rund::node::CompletionPool::park(ordered, order_c));
  TEST_ASSERT(rund::node::CompletionPool::unpark(ordered, order_b));
  TEST_ASSERT(rund::node::CompletionPool::terminate(
      ordered, rund::ReasonCode::TaskFailed));
  TEST_ASSERT(order.count == 2u);
  TEST_ASSERT(order.values[0] == 1u);
  TEST_ASSERT(order.values[1] == 3u);
  TEST_ASSERT(order_a.state.load(std::memory_order_acquire) ==
              rund::node::CompletionWaiterState::Idle);
  TEST_ASSERT(order_b.state.load(std::memory_order_acquire) ==
              rund::node::CompletionWaiterState::Idle);
  TEST_ASSERT(order_c.state.load(std::memory_order_acquire) ==
              rund::node::CompletionWaiterState::Idle);
  rund::node::CompletionPool::release(ordered);
  return 0;
}
