#include <poll.h>
#include <unistd.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <rund/host.hpp>
#include <rund/session.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>
#include <vector>

#include "../../../../../../src/runtime/reactor/diagnostics.hpp"
#include "../../../../../../src/runtime/reactor/readiness/handle.hpp"
#include "../../../../../../src/runtime/task/scheduler/reactor/backend.hpp"
#include "../../../../../../src/runtime/task/scheduler/reactor/backlog.hpp"
#include "../../../../../../src/runtime/task/scheduler/reactor/generation.hpp"
#include "../../../../../../src/runtime/task/scheduler/reactor/registry.hpp"
#include "../await.hpp"
#include "test/assert.hpp"

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

} // namespace

int RunRuntimeTaskReactorBudgetBacklogContract() {
  constexpr std::size_t kTasks = 256u;
  constexpr std::uint32_t kBudget = 32u;
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
                  .task_capacity = 320u,
                  .ready_queue_capacity = 320u,
                  .reactor_wait_capacity = 320u,
                  .reactor_ready_budget = kBudget,
                  .observation_capacity = 1024u,
                  .host_event_capacity = 1024u,
              },
      },
      [&] {
        scoped = rund::task::scope([&] {
          for (std::size_t index = 0u; index < kTasks; ++index) {
            (void)rund::task::spawn(
                "budget-backlog-reader",
                rund::node::test_contract::reactor::AwaitReadable(
                    ready_fds[index].view(), &ready[index]));
          }
          (void)rund::task::spawn("budget-backlog-writer", [&] {
            const char byte = 'q';
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
  TEST_ASSERT(stats.ready_backlog_pushes > 0u);
  TEST_ASSERT(stats.ready_backlog_drains > 0u);
  TEST_ASSERT(stats.ready_expansion_scan_steps <= kTasks + kBudget);
  TEST_ASSERT(stats.ready_backlog_scan_steps_avoided >= kTasks - kBudget);

  rund::node::ResetReactorBackendStats();
  rund::node::ReactorRuntime stale_reactor{};
  TEST_ASSERT(rund::node::ReactorRegistryPrepare(stale_reactor, 4u));
  const rund::node::ReactorWait stale_wait{
      .task_id = 11u,
      .wait_id = 21u,
      .host_handle_id = 31u,
      .fd_generation = 1u,
      .fd = rund::node::ReactorHandleFromPublic(77),
      .interest = rund::node::ReactorInterest::Read};
  const rund::node::ReactorReady stale_ready{
      .wait_id = stale_wait.wait_id,
      .task_id = stale_wait.task_id,
      .fd = stale_wait.fd,
      .interest = rund::node::ReactorInterest::Read,
      .events = rund::node::ReactorEvent::Read};
  const rund::node::ReactorReady live_ready{
      .wait_id = 22u,
      .task_id = 12u,
      .fd = rund::node::ReactorHandleFromPublic(78),
      .interest = rund::node::ReactorInterest::Read,
      .events = rund::node::ReactorEvent::Read};
  TEST_ASSERT(rund::node::ReactorRegistryAddWait(stale_reactor, stale_wait));
  stale_reactor.ready_backlog.push_back(stale_ready);
  stale_reactor.ready_backlog.push_back(live_ready);
  std::vector<rund::node::ReactorWait> stale_waits{};
  TEST_ASSERT(rund::node::ReactorGenerationCollectStaleWaits(
      stale_reactor, stale_wait.fd, 2u, stale_waits));
  rund::node::ReactorBacklogRemoveFd(stale_reactor, stale_wait.fd);
  const rund::node::ReactorBackendStats stale_stats =
      rund::node::ReactorBackendStatsSnapshot();
  TEST_ASSERT(stale_waits.size() == 1u);
  TEST_ASSERT(stale_waits[0].wait_id == stale_wait.wait_id);
  TEST_ASSERT(rund::node::ReactorRegistrySize(stale_reactor) == 1u);
  TEST_ASSERT(stale_reactor.ready_backlog.size() == 1u);
  TEST_ASSERT(stale_reactor.ready_backlog[0].fd == live_ready.fd);
  TEST_ASSERT(stale_stats.ready_backlog_invalidations == 1u);

  rund::node::ResetReactorBackendStats();
  rund::node::ReactorRuntime close_reactor{};
  close_reactor.ready_backlog.push_back(stale_ready);
  close_reactor.ready_backlog.push_back(live_ready);
  rund::node::ReactorCloseRuntime(close_reactor);
  const rund::node::ReactorBackendStats close_stats =
      rund::node::ReactorBackendStatsSnapshot();
  TEST_ASSERT(close_reactor.ready_backlog.empty());
  TEST_ASSERT(close_stats.ready_backlog_invalidations == 2u);
  return 0;
}
