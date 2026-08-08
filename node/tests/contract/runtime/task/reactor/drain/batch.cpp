#include "../await.hpp"
#include "test/assert.hpp"

#include <rund/host.hpp>
#include <rund/session.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include "../../../../../../src/runtime/reactor/diagnostics.hpp"
#include "../../../../../../src/runtime/reactor/readiness/handle.hpp"
#include "../../../../../../src/runtime/reactor/readiness/mask.hpp"
#include "../../../../../../src/runtime/task/scheduler/reactor/drain/batch.hpp"
#include "../../../../../../src/runtime/task/scheduler/reactor/registry.hpp"
#include "../../coroutine/allocation.hpp"

#include <type_traits>
#include <vector>

#include <unistd.h>

namespace {

struct PipeCleanup {
  int read_fd = -1;
  int write_fd = -1;

  ~PipeCleanup() {
    if (read_fd >= 0) {
      static_cast<void>(::close(read_fd));
    }
    if (write_fd >= 0) {
      static_cast<void>(::close(write_fd));
    }
  }

  PipeCleanup() = default;
  PipeCleanup(const PipeCleanup &) = delete;
  PipeCleanup &operator=(const PipeCleanup &) = delete;
  PipeCleanup(PipeCleanup &&other) noexcept
      : read_fd(other.read_fd), write_fd(other.write_fd) {
    other.read_fd = -1;
    other.write_fd = -1;
  }
  PipeCleanup &operator=(PipeCleanup &&other) noexcept {
    if (this == &other) {
      return *this;
    }
    this->~PipeCleanup();
    read_fd = other.read_fd;
    write_fd = other.write_fd;
    other.read_fd = -1;
    other.write_fd = -1;
    return *this;
  }
};

[[nodiscard]] PipeCleanup MakePipe() {
  int fds[2] = {-1, -1};
  if (::pipe(fds) != 0) {
    return {};
  }
  PipeCleanup pipe{};
  pipe.read_fd = fds[0];
  pipe.write_fd = fds[1];
  return pipe;
}

[[nodiscard]] rund::node::ReactorReady
ReadyFor(const rund::node::ReactorWait &wait) noexcept {
  return rund::node::ReactorReady{
      .wait_id = wait.wait_id,
      .task_id = wait.task_id,
      .fd = wait.fd,
      .interest = wait.interest,
      .events = rund::node::ReactorEventsForInterest(wait.interest),
  };
}

void VerifyDrainBatchStates() {
  using namespace rund::node;

  static_assert(!std::is_aggregate_v<ReactorDrainBatch>);
  static_assert(!std::is_default_constructible_v<ReactorDrainBatch>);
  static_assert(std::is_trivially_copyable_v<ReactorDrainBatch>);

  ReactorRuntime rejected_reactor{};
  rejected_reactor.previous_interest_scratch.reserve(1u);
  const std::size_t rejected_count =
      rejected_reactor.previous_interest_scratch.capacity() + 1u;
  TEST_ASSERT(ReactorRegistryPrepare(rejected_reactor, rejected_count));
  std::vector<ReactorWait> rejected_waits(rejected_count);
  std::vector<ReactorReady> rejected_ready(rejected_count);
  for (std::size_t index = 0u; index < rejected_count; ++index) {
    rejected_waits[index] = ReactorWait{
        .task_id = index + 1u,
        .wait_id = index + 11u,
        .fd_generation = 1u,
        .fd = ReactorHandleFromPublic(static_cast<int>(index + 101u)),
        .interest = ReactorInterest::Read,
    };
    rejected_ready[index] = ReadyFor(rejected_waits[index]);
    TEST_ASSERT(
        ReactorRegistryAddWait(rejected_reactor, rejected_waits[index]));
  }
  rejected_reactor.drain_ready_scratch.reserve(rejected_count);
  rejected_reactor.removed_wait_scratch.reserve(rejected_count);
  rejected_reactor.changes.reserve(rejected_count);
  runtime_task_allocation::FailNext();
  const ReactorDrainBatch rejected =
      ReactorBuildDrainBatch(rejected_reactor, rejected_ready);
  TEST_ASSERT(rejected.disposition() ==
              ReactorDrainBatchDisposition::Rejected);
  TEST_ASSERT(rejected.as_failed().disposition() ==
              ReactorDrainBatchDisposition::Rejected);
  TEST_ASSERT(ReactorRegistrySize(rejected_reactor) == rejected_count);
  for (std::size_t index = 0u; index < rejected_count; ++index) {
    TEST_ASSERT(ReactorRegistryWaitAt(rejected_reactor, index).wait_id ==
                rejected_waits[index].wait_id);
    TEST_ASSERT(ReactorRegistryInterestForFd(
                    rejected_reactor, rejected_waits[index].fd) ==
                rejected_waits[index].interest);
  }
  TEST_ASSERT(rejected_reactor.removed_wait_scratch.empty());
  TEST_ASSERT(rejected_reactor.previous_interest_scratch.empty());
  TEST_ASSERT(rejected_reactor.changes.empty());

  const ReactorWait first_wait{
      .task_id = 11u,
      .wait_id = 21u,
      .fd_generation = 1u,
      .fd = ReactorHandleFromPublic(31),
      .interest = ReactorInterest::Read,
  };
  const ReactorWait second_wait{
      .task_id = 12u,
      .wait_id = 22u,
      .fd_generation = 1u,
      .fd = ReactorHandleFromPublic(32),
      .interest = ReactorInterest::Write,
  };
  const std::vector<ReactorReady> complete_ready{
      ReadyFor(first_wait),
      ReadyFor(second_wait),
  };
  ReactorRuntime complete_reactor{};
  TEST_ASSERT(ReactorRegistryPrepare(complete_reactor, 2u));
  TEST_ASSERT(ReactorRegistryAddWait(complete_reactor, first_wait));
  TEST_ASSERT(ReactorRegistryAddWait(complete_reactor, second_wait));
  const ReactorDrainBatch complete =
      ReactorBuildDrainBatch(complete_reactor, complete_ready);
  TEST_ASSERT(complete.disposition() ==
              ReactorDrainBatchDisposition::Complete);
  TEST_ASSERT(complete.ready().size() == 2u);
  TEST_ASSERT(complete.removed_waits().size() == 2u);
  TEST_ASSERT(complete.ready()[0].wait_id == first_wait.wait_id);
  TEST_ASSERT(complete.removed_waits()[0].wait_id == first_wait.wait_id);
  TEST_ASSERT(complete.ready()[1].wait_id == second_wait.wait_id);
  TEST_ASSERT(complete.removed_waits()[1].wait_id == second_wait.wait_id);
  TEST_ASSERT(ReactorRegistryEmpty(complete_reactor));

  const ReactorDrainBatch lowered = complete.as_failed();
  TEST_ASSERT(lowered.disposition() == ReactorDrainBatchDisposition::Failed);
  TEST_ASSERT(&lowered.ready() == &complete.ready());
  TEST_ASSERT(&lowered.removed_waits() == &complete.removed_waits());

  ReactorRuntime partial_reactor{};
  TEST_ASSERT(ReactorRegistryPrepare(partial_reactor, 2u));
  TEST_ASSERT(ReactorRegistryAddWait(partial_reactor, first_wait));
  TEST_ASSERT(ReactorRegistryAddWait(partial_reactor, second_wait));
  const std::vector<ReactorReady> partial_ready{
      ReadyFor(first_wait),
      ReactorReady{
          .wait_id = 999u,
          .task_id = 999u,
          .fd = first_wait.fd,
          .interest = first_wait.interest,
      },
      ReadyFor(second_wait),
  };
  const ReactorDrainBatch partial =
      ReactorBuildDrainBatch(partial_reactor, partial_ready);
  TEST_ASSERT(partial.disposition() == ReactorDrainBatchDisposition::Failed);
  TEST_ASSERT(partial.ready().size() == 1u);
  TEST_ASSERT(partial.removed_waits().size() == 1u);
  TEST_ASSERT(partial.ready()[0].wait_id == first_wait.wait_id);
  TEST_ASSERT(partial.removed_waits()[0].wait_id == first_wait.wait_id);
  const ReactorDrainBatch failed_again = partial.as_failed();
  TEST_ASSERT(failed_again.disposition() ==
              ReactorDrainBatchDisposition::Failed);
  TEST_ASSERT(&failed_again.ready() == &partial.ready());
  TEST_ASSERT(&failed_again.removed_waits() == &partial.removed_waits());
  TEST_ASSERT(ReactorRegistrySize(partial_reactor) == 1u);
  TEST_ASSERT(ReactorRegistryWaitAt(partial_reactor, 0u).wait_id ==
              second_wait.wait_id);
}

} // namespace

int RunRuntimeTaskReactorBatchDrainContract() {
  VerifyDrainBatchStates();

  PipeCleanup first = MakePipe();
  PipeCleanup second = MakePipe();
  TEST_ASSERT(first.read_fd >= 0 && first.write_fd >= 0);
  TEST_ASSERT(second.read_fd >= 0 && second.write_fd >= 0);
  rund::host::io::Fd first_fd =
      rund::host::io::take_native_fd(::dup(first.read_fd));
  rund::host::io::Fd second_fd =
      rund::host::io::take_native_fd(::dup(second.read_fd));
  TEST_ASSERT(first_fd && second_fd);

  rund::node::ResetReactorBackendStats();
  std::vector<int> wake_order{};
  rund::task::IoResult first_ready{};
  rund::task::IoResult second_ready{};
  rund::task::Status joined{};
  bool writes_ok = false;

  const rund::Session::Result report = rund::run(
      rund::SessionConfig{
          .workers = 1u,
          .scheduler =
              {
                  .task_capacity = 4u,
                  .ready_queue_capacity = 4u,
                  .reactor_wait_capacity = 4u,
                  .observation_capacity = 8u,
                  .host_event_capacity = 8u,
              },
      },
      [&] {
        const rund::task::Handle first_reader =
            rund::task::spawn("reactor-batch-first",
                              rund::node::test_contract::reactor::AwaitReadable(
                                  first_fd.view(), &first_ready,
                                  [&] { wake_order.push_back(1); }));
        const rund::task::Handle second_reader =
            rund::task::spawn("reactor-batch-second",
                              rund::node::test_contract::reactor::AwaitReadable(
                                  second_fd.view(), &second_ready,
                                  [&] { wake_order.push_back(2); }));
        const rund::task::Handle writer =
            rund::task::spawn("reactor-batch-writer", [&] {
              const char a = 'a';
              const char b = 'b';
              const bool wrote_first =
                  ::write(first.write_fd, &a, 1u) == 1;
              const bool wrote_second =
                  ::write(second.write_fd, &b, 1u) == 1;
              writes_ok = wrote_first && wrote_second;
            });
        joined = rund::task::join(first_reader, second_reader, writer);
      });

  const rund::node::ReactorBackendStats stats =
      rund::node::ReactorBackendStatsSnapshot();

  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(writes_ok);
  TEST_ASSERT(first_ready.ok());
  TEST_ASSERT(second_ready.ok());
  TEST_ASSERT(wake_order.size() == 2u);
  TEST_ASSERT(wake_order[0] == 1);
  TEST_ASSERT(wake_order[1] == 2);
  TEST_ASSERT(stats.max_ready_batch >= 2u);
  TEST_ASSERT(stats.poll_calls >= 1u);
  return 0;
}
