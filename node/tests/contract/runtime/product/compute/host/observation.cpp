#include "local.hpp"

#include "src/runtime/compute/local.hpp"
#include "src/runtime/task/scheduler/task/completion.hpp"

#include <chrono>
#include <thread>

namespace runtime_compute_host_detail {
namespace {
struct Pending final {
  rund::node::CompletionPool pool;
  rund::node::compute_detail::TaskState state;
  rund::node::CompletionLease lease;
  ~Pending() {
    const auto observer = state.completion;
    if (observer.release != nullptr)
      observer.release(observer.authority, observer.slot, observer.generation);
    rund::node::CompletionPool::release(lease);
  }
  bool initialize(const rund::compute::Status status) {
    using namespace rund;
    if (!pool.configure(node::CompletionLimits{.capacity = 1u}))
      return false;
    lease = pool.claim();
    if (!lease ||
        !node::CompletionPool::transition(lease, task::Phase::Ready) ||
        !node::CompletionPool::transition(lease, task::Phase::Running) ||
        !node::CompletionPool::transition(lease, task::Phase::Committing))
      return false;
    state.completion = pool.observe_ref(lease);
    state.status = status;
    state.submitted = true;
    state.backend_submitted.store(true, std::memory_order_release);
    state.terminal_phase.store(node::compute_detail::TerminalPhase::Complete,
                               std::memory_order_release);
    return state.completion.poll != nullptr;
  }
};
} // namespace

bool CheckTimedSubmissionObservation() {
  using namespace rund;
  Pending pending;
  if (!pending.initialize(compute::Status::success()))
    return false;
  constexpr auto timeout = std::chrono::milliseconds{20};
  const auto started = std::chrono::steady_clock::now();
  const auto polled = node::WaitSubmission(pending.state, timeout);
  const auto elapsed = std::chrono::steady_clock::now() - started;
  // This is the timeout contract's lower bound, with no performance upper
  // bound. The old Compute-only predicate returned immediately in this gap.
  if (polled.terminal() || polled.code != ReasonCode::Ok || elapsed < timeout)
    return false;
  bool published = false;
  std::thread publisher([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
    published =
        static_cast<bool>(node::CompletionPool::complete(pending.lease));
  });
  const auto ready =
      node::WaitSubmission(pending.state, std::chrono::seconds{5});
  publisher.join();
  if (!published || !ready.terminal() || ready.code != ReasonCode::Ok ||
      !node::WaitSubmission(pending.state, std::chrono::nanoseconds::zero())
           .terminal())
    return false;
  auto foreign = pending.state.completion;
  foreign.poll = nullptr;
  if (node::CompletionPool::wait_for(foreign, timeout).code !=
      ReasonCode::TaskHandleStale)
    return false;
  Pending failed;
  if (!failed.initialize(
          compute::Status::fail(compute::Reason::BackendFailed)) ||
      !node::CompletionPool::terminate(failed.lease, ReasonCode::TaskFailed))
    return false;
  const auto failure =
      node::WaitSubmission(failed.state, std::chrono::seconds{5});
  return failure.terminal() && failure.code == ReasonCode::TaskFailed;
}
} // namespace runtime_compute_host_detail
