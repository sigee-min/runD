#include "await.hpp"
#include "test/assert.hpp"

#include <rund/host.hpp>
#include <rund/session.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include "../../../../../src/host/io/access.hpp"
#include "../../../../../src/runtime/reactor/diagnostics.hpp"
#include "../../../../../src/runtime/reactor/readiness/handle.hpp"
#include "../../../../../src/runtime/task/scheduler/reactor/budget.hpp"
#include "../coroutine/allocation.hpp"

#include <array>
#include <cstddef>
#include <type_traits>
#include <vector>

#include <unistd.h>

namespace {

struct PipePair {
  int read_fd = -1;
  int write_fd = -1;
  ~PipePair() {
    if (read_fd >= 0) {
      static_cast<void>(::close(read_fd));
    }
    if (write_fd >= 0) {
      static_cast<void>(::close(write_fd));
    }
  }
};

[[nodiscard]] bool MakePipe(PipePair &pipe) {
  int fds[2] = {-1, -1};
  if (::pipe(fds) != 0) {
    return false;
  }
  pipe.read_fd = fds[0];
  pipe.write_fd = fds[1];
  return true;
}

void VerifyInvalidFdBudgetRetirement() {
  PipePair pipe{};
  PipePair suffix_pipe{};
  TEST_ASSERT(MakePipe(pipe));
  TEST_ASSERT(MakePipe(suffix_pipe));
  rund::host::io::Fd fd =
      rund::host::io::take_native_fd(::dup(pipe.read_fd));
  rund::host::io::Fd suffix_fd =
      rund::host::io::take_native_fd(::dup(suffix_pipe.read_fd));
  TEST_ASSERT(fd);
  TEST_ASSERT(suffix_fd);
  const rund::host::io::FdView view = fd.view();
  const rund::host::io::FdView suffix_view = suffix_fd.view();
  const int native = rund::host::io::detail::Access::native(view);
  const int suffix_native =
      rund::host::io::detail::Access::native(suffix_view);

  rund::task::IoResult read_result{};
  rund::task::IoResult write_result{};
  rund::task::IoResult suffix_result{};
  rund::task::Status closer_yield{};
  rund::task::Status joined{};
  bool read_started = false;
  bool write_started = false;
  bool suffix_started = false;
  bool raw_closed = false;
  bool suffix_raw_closed = false;
  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_capacity = 8u,
                  .ready_queue_capacity = 8u,
                  .reactor_wait_capacity = 8u,
                  .reactor_ready_budget = 1u,
                  .observation_capacity = 32u,
                  .host_event_capacity = 32u,
              },
      },
      [&] {
        auto read = [&]() -> rund::task::Task<void> {
          read_started = true;
          read_result = co_await rund::host::io::readable(view);
        };
        auto write = [&]() -> rund::task::Task<void> {
          write_started = true;
          write_result = co_await rund::host::io::writable(view);
        };
        auto suffix = [&]() -> rund::task::Task<void> {
          suffix_started = true;
          suffix_result = co_await rund::host::io::readable(suffix_view);
        };
        auto close = [&]() -> rund::task::Task<void> {
          closer_yield = co_await rund::task::yield();
          raw_closed = ::close(native) == 0;
          suffix_raw_closed = ::close(suffix_native) == 0;
        };
        const rund::task::Handle reader =
            rund::task::spawn("invalid-budget-reader", read());
        const rund::task::Handle writer =
            rund::task::spawn("invalid-budget-writer", write());
        const rund::task::Handle suffix_reader =
            rund::task::spawn("invalid-budget-suffix", suffix());
        const rund::task::Handle closer =
            rund::task::spawn("invalid-budget-closer", close());
        joined = rund::task::join(reader, writer, suffix_reader, closer);
      });

  static_cast<void>(fd.close());
  static_cast<void>(suffix_fd.close());
  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(closer_yield.ok());
  TEST_ASSERT(read_started);
  TEST_ASSERT(write_started);
  TEST_ASSERT(suffix_started);
  TEST_ASSERT(raw_closed);
  TEST_ASSERT(suffix_raw_closed);
  TEST_ASSERT(!read_result.ok());
  TEST_ASSERT(read_result.code() == rund::ReasonCode::IoFdInvalid);
  TEST_ASSERT(!write_result.ok());
  TEST_ASSERT(write_result.code() == rund::ReasonCode::IoFdInvalid);
  TEST_ASSERT(!suffix_result.ok());
  TEST_ASSERT(suffix_result.code() == rund::ReasonCode::IoFdInvalid);
}

} // namespace

int RunRuntimeTaskReactorBudgetContract() {
  static_assert(!std::is_aggregate_v<rund::node::ReactorBudgetSelection>);
  static_assert(
      std::is_trivially_copyable_v<rund::node::ReactorBudgetSelection>);

  rund::node::ReactorRuntime selection_reactor{};
  const std::vector<rund::node::ReactorReady> full_ready{
      rund::node::ReactorReady{.wait_id = 1u},
      rund::node::ReactorReady{.wait_id = 2u},
  };
  const rund::node::ReactorBudgetSelection zero_selection =
      rund::node::ReactorBudgetSelect(selection_reactor, full_ready, 0u);
  TEST_ASSERT(!zero_selection.ok());
  TEST_ASSERT(zero_selection.consumed() == 0u);

  const rund::node::ReactorBudgetSelection full_selection =
      rund::node::ReactorBudgetSelect(selection_reactor, full_ready,
                                      full_ready.size());
  TEST_ASSERT(full_selection.ok());
  TEST_ASSERT(&full_selection.ready() == &full_ready);
  TEST_ASSERT(full_selection.consumed() == full_ready.size());

  selection_reactor.budget_ready_scratch.reserve(1u);
  const std::size_t prefix_budget =
      selection_reactor.budget_ready_scratch.capacity() + 1u;
  std::vector<rund::node::ReactorReady> prefixed_ready(prefix_budget + 1u);
  for (std::size_t index = 0u; index < prefixed_ready.size(); ++index) {
    prefixed_ready[index].wait_id = index + 1u;
  }
  runtime_task_allocation::FailNext();
  const rund::node::ReactorBudgetSelection failed_selection =
      rund::node::ReactorBudgetSelect(selection_reactor, prefixed_ready,
                                      prefix_budget);
  TEST_ASSERT(!failed_selection.ok());
  TEST_ASSERT(failed_selection.consumed() == 0u);
  TEST_ASSERT(selection_reactor.budget_ready_scratch.empty());

  const rund::node::ReactorBudgetSelection prefix_selection =
      rund::node::ReactorBudgetSelect(selection_reactor, prefixed_ready,
                                      prefix_budget);
  TEST_ASSERT(prefix_selection.ok());
  TEST_ASSERT(&prefix_selection.ready() ==
              &selection_reactor.budget_ready_scratch);
  TEST_ASSERT(prefix_selection.consumed() == prefix_budget);
  for (std::size_t index = 0u; index < prefix_selection.consumed(); ++index) {
    TEST_ASSERT(prefix_selection.ready()[index].wait_id == index + 1u);
  }
  selection_reactor.budget_ready_scratch.push_back(
      prefixed_ready[prefix_budget]);
  TEST_ASSERT(prefix_selection.consumed() == prefix_budget + 1u);
  TEST_ASSERT(prefix_selection.ready().back().wait_id == prefix_budget + 1u);

  runtime_task_allocation::Start();
  const rund::node::ReactorBudgetSelection warm_selection =
      rund::node::ReactorBudgetSelect(selection_reactor, prefixed_ready,
                                      prefix_budget);
  runtime_task_allocation::Stop();
  TEST_ASSERT(warm_selection.ok());
  TEST_ASSERT(warm_selection.consumed() == prefix_budget);
  TEST_ASSERT(runtime_task_allocation::Count() == 0u);

  const rund::node::ReactorHandle invalid_fd =
      rund::node::ReactorHandleFromPublic(11);
  const std::vector<rund::node::ReactorReady> same_invalid_fd{
      rund::node::ReactorReady{
          .wait_id = 1u,
          .fd = invalid_fd,
          .disposition = rund::node::ReactorReadyDisposition::Invalid},
      rund::node::ReactorReady{
          .wait_id = 2u,
          .fd = invalid_fd,
          .disposition = rund::node::ReactorReadyDisposition::Invalid}};
  TEST_ASSERT(rund::node::ReactorBudgetExtendInvalidFdPrefix(same_invalid_fd,
                                                             1u) == 2u);
  std::vector<rund::node::ReactorReady> distinct_invalid_fds = same_invalid_fd;
  distinct_invalid_fds[1].fd = rund::node::ReactorHandleFromPublic(12);
  TEST_ASSERT(rund::node::ReactorBudgetExtendInvalidFdPrefix(
                  distinct_invalid_fds, 1u) == 1u);
  const rund::node::ReactorHandle chained_fd =
      rund::node::ReactorHandleFromPublic(13);
  std::vector<rund::node::ReactorReady> chained_invalid_fds = same_invalid_fd;
  chained_invalid_fds.insert(
      chained_invalid_fds.begin() + 1,
      rund::node::ReactorReady{
          .wait_id = 3u,
          .fd = chained_fd,
          .disposition = rund::node::ReactorReadyDisposition::Invalid});
  chained_invalid_fds.push_back(rund::node::ReactorReady{
      .wait_id = 4u,
      .fd = chained_fd,
      .disposition = rund::node::ReactorReadyDisposition::Invalid});
  TEST_ASSERT(rund::node::ReactorBudgetExtendInvalidFdPrefix(
                  chained_invalid_fds, 1u) == 4u);
  std::vector<rund::node::ReactorReady> ready_prefix = same_invalid_fd;
  ready_prefix[0].disposition = rund::node::ReactorReadyDisposition::Ready;
  TEST_ASSERT(
      rund::node::ReactorBudgetExtendInvalidFdPrefix(ready_prefix, 1u) == 1u);

  VerifyInvalidFdBudgetRetirement();

  constexpr std::size_t kTasks = 16u;
  constexpr std::uint32_t kBudget = 4u;
  std::array<PipePair, kTasks> pipes{};
  for (PipePair &pipe : pipes) {
    TEST_ASSERT(MakePipe(pipe));
  }
  std::array<rund::host::io::Fd, kTasks> ready_fds{};
  for (std::size_t index = 0u; index < kTasks; ++index) {
    ready_fds[index] =
        rund::host::io::take_native_fd(::dup(pipes[index].read_fd));
    TEST_ASSERT(ready_fds[index]);
  }

  rund::node::ResetReactorBackendStats();
  std::array<rund::task::IoResult, kTasks> ready{};
  rund::task::Status scoped{};
  bool writes_ok = true;
  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_capacity = 32u,
                  .ready_queue_capacity = 32u,
                  .reactor_wait_capacity = 32u,
                  .reactor_ready_budget = kBudget,
                  .observation_capacity = 128u,
                  .host_event_capacity = 128u,
              },
      },
      [&] {
        scoped = rund::task::scope([&] {
          for (std::size_t index = 0u; index < kTasks; ++index) {
            (void)rund::task::spawn(
                "budget-reader",
                rund::node::test_contract::reactor::AwaitReadable(
                    ready_fds[index].view(), &ready[index]));
          }
          (void)rund::task::spawn("budget-writer", [&] {
            const char byte = 'b';
            for (PipePair &pipe : pipes) {
              writes_ok =
                  (::write(pipe.write_fd, &byte, 1u) == 1) && writes_ok;
            }
          });
        });
      });

  const rund::node::ReactorBackendStats stats =
      rund::node::ReactorBackendStatsSnapshot();
  TEST_ASSERT(report.ok());
  TEST_ASSERT(scoped.ok());
  TEST_ASSERT(writes_ok);
  for (const rund::task::IoResult &result : ready) {
    TEST_ASSERT(result.ok());
  }
  TEST_ASSERT(stats.max_ready_batch <= kBudget);
  TEST_ASSERT(stats.ready_budget_deferrals > 0u);
  TEST_ASSERT(stats.scratch_ready_reuses > 0u);
  return 0;
}
