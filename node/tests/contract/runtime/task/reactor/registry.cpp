#include "await.hpp"
#include "test/assert.hpp"

#include <rund/host.hpp>
#include <rund/session.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include "../../../../../src/runtime/reactor/diagnostics.hpp"
#include "../../../../../src/runtime/reactor/readiness/handle.hpp"
#include "../../../../../src/runtime/reactor/readiness/mask.hpp"
#include "../../../../../src/runtime/task/scheduler/reactor/registry.hpp"

#include <array>
#include <vector>

#include <unistd.h>

namespace {

struct PipeCleanup {
  int read_fd = -1;
  int write_fd = -1;

  ~PipeCleanup() {
    if (read_fd >= 0)
      static_cast<void>(::close(read_fd));
    if (write_fd >= 0)
      static_cast<void>(::close(write_fd));
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
    if (this == &other)
      return *this;
    if (read_fd >= 0)
      static_cast<void>(::close(read_fd));
    if (write_fd >= 0)
      static_cast<void>(::close(write_fd));
    read_fd = other.read_fd;
    write_fd = other.write_fd;
    other.read_fd = -1;
    other.write_fd = -1;
    return *this;
  }
};

[[nodiscard]] PipeCleanup MakePipe() {
  int fds[2] = {-1, -1};
  if (::pipe(fds) != 0)
    return {};
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

void VerifyRegistryBatchProjection() {
  using rund::node::ReactorHandleFromPublic;
  using rund::node::ReactorInterest;
  using rund::node::ReactorWait;

  static_assert(sizeof(ReactorWait) == 88u);
  static_assert(sizeof(rund::node::ReactorWaitSlot) +
                    sizeof(std::uint32_t) <=
                sizeof(ReactorWait) + 3u * sizeof(std::uint64_t));

  const ReactorWait first{
      .task_id = 12u,
      .wait_id = 2u,
      .fd = ReactorHandleFromPublic(9),
      .interest = ReactorInterest::Read,
  };
  const ReactorWait second{
      .task_id = 14u,
      .wait_id = 4u,
      .fd = ReactorHandleFromPublic(9),
      .interest = ReactorInterest::Write,
  };
  const ReactorWait third{
      .task_id = 16u,
      .wait_id = 6u,
      .fd = ReactorHandleFromPublic(7),
      .interest = ReactorInterest::Read,
  };
  rund::node::ReactorRuntime reactor{};
  TEST_ASSERT(rund::node::ReactorRegistryPrepare(reactor, 8u));
  const rund::node::ReactorWaitSlot *const slots =
      reactor.registry.slots.data();
  const std::uint32_t *const order = reactor.registry.order.data();
  const rund::node::ReactorFdState *const fds = reactor.registry.fds.data();
  const std::uint32_t *const free_slots =
      reactor.registry.free_slots.data();
  const std::size_t slot_capacity = reactor.registry.slots.capacity();
  const std::size_t order_capacity = reactor.registry.order.capacity();
  const std::size_t fd_capacity = reactor.registry.fds.capacity();
  const std::size_t free_capacity = reactor.registry.free_slots.capacity();
  TEST_ASSERT(!rund::node::ReactorRegistryAddWait(
      reactor, ReactorWait{.task_id = 1u,
                           .wait_id = 0u,
                           .fd = ReactorHandleFromPublic(7),
                           .interest = ReactorInterest::Read}));
  TEST_ASSERT(rund::node::ReactorRegistryAddWait(reactor, third));
  TEST_ASSERT(rund::node::ReactorRegistryAddWait(reactor, first));
  TEST_ASSERT(rund::node::ReactorRegistryAddWait(reactor, second));
  TEST_ASSERT(rund::node::ReactorRegistrySize(reactor) == 3u);
  TEST_ASSERT(rund::node::ReactorRegistryWaitAt(reactor, 0u).wait_id ==
              first.wait_id);
  TEST_ASSERT(rund::node::ReactorRegistryWaitAt(reactor, 1u).wait_id ==
              second.wait_id);
  TEST_ASSERT(rund::node::ReactorRegistryWaitAt(reactor, 2u).wait_id ==
              third.wait_id);

  const rund::node::ReactorFdState *const first_fd =
      rund::node::ReactorRegistryFindFd(reactor, first.fd);
  TEST_ASSERT(first_fd != nullptr);
  TEST_ASSERT(first_fd->wait_count == 2u);
  const std::uint32_t first_slot =
      rund::node::ReactorRegistryFirstWait(reactor, first.fd);
  const std::uint32_t second_slot =
      rund::node::ReactorRegistryNextWait(reactor, first_slot);
  TEST_ASSERT(first_slot != rund::node::kNoReactorSlot);
  TEST_ASSERT(second_slot != rund::node::kNoReactorSlot);
  TEST_ASSERT(rund::node::ReactorRegistryNextWait(reactor, second_slot) ==
              rund::node::kNoReactorSlot);
  TEST_ASSERT(rund::node::ReactorRegistrySlotWait(reactor, first_slot)
                  ->wait_id == first.wait_id);
  TEST_ASSERT(rund::node::ReactorRegistrySlotWait(reactor, second_slot)
                  ->wait_id == second.wait_id);
  TEST_ASSERT(rund::node::ReactorRegistryInterestForFd(reactor, first.fd) ==
              (ReactorInterest::Read | ReactorInterest::Write));

  std::vector<rund::node::ReactorReady> ordered{
      ReadyFor(first),
      rund::node::ReactorReady{
          .wait_id = 3u,
          .task_id = 13u,
          .fd = first.fd,
          .interest = ReactorInterest::Read,
      },
      ReadyFor(second),
  };
  std::vector<ReactorWait> removed{};
  std::vector<rund::node::ReactorFdPreviousInterest> affected{};
  TEST_ASSERT(!rund::node::ReactorRegistryRemoveReadyBatch(
      reactor, ordered, removed, affected));
  TEST_ASSERT(removed.size() == 1u);
  TEST_ASSERT(affected.size() == 1u);
  TEST_ASSERT(affected[0].fd == first.fd);
  TEST_ASSERT(affected[0].interest ==
              (ReactorInterest::Read | ReactorInterest::Write));
  TEST_ASSERT(removed[0].wait_id == first.wait_id);
  TEST_ASSERT(rund::node::ReactorRegistrySize(reactor) == 2u);
  TEST_ASSERT(rund::node::ReactorRegistryWaitAt(reactor, 0u).wait_id ==
              second.wait_id);
  TEST_ASSERT(rund::node::ReactorRegistryWaitAt(reactor, 1u).wait_id ==
              third.wait_id);
  TEST_ASSERT(rund::node::ReactorRegistryInterestForFd(reactor, first.fd) ==
              ReactorInterest::Write);

  ordered = {ReadyFor(second), ReadyFor(third)};
  TEST_ASSERT(rund::node::ReactorRegistryRemoveReadyBatch(
      reactor, ordered, removed, affected));
  TEST_ASSERT(removed.size() == 2u);
  TEST_ASSERT(affected.size() == 2u);
  TEST_ASSERT(affected[0].fd == second.fd);
  TEST_ASSERT(affected[0].interest == ReactorInterest::Write);
  TEST_ASSERT(affected[1].fd == third.fd);
  TEST_ASSERT(affected[1].interest == ReactorInterest::Read);
  TEST_ASSERT(removed[0].wait_id == second.wait_id);
  TEST_ASSERT(removed[1].wait_id == third.wait_id);
  TEST_ASSERT(rund::node::ReactorRegistryEmpty(reactor));
  TEST_ASSERT(rund::node::ReactorRegistryFindFd(reactor, second.fd)
                  ->wait_count == 0u);
  TEST_ASSERT(rund::node::ReactorRegistryFindFd(reactor, third.fd)
                  ->wait_count == 0u);
  TEST_ASSERT(reactor.registry.slots.data() == slots);
  TEST_ASSERT(reactor.registry.order.data() == order);
  TEST_ASSERT(reactor.registry.fds.data() == fds);
  TEST_ASSERT(reactor.registry.free_slots.data() == free_slots);
  TEST_ASSERT(reactor.registry.slots.capacity() == slot_capacity);
  TEST_ASSERT(reactor.registry.order.capacity() == order_capacity);
  TEST_ASSERT(reactor.registry.fds.capacity() == fd_capacity);
  TEST_ASSERT(reactor.registry.free_slots.capacity() == free_capacity);
}

} // namespace

int RunRuntimeTaskReactorRegistryContract() {
  VerifyRegistryBatchProjection();
  constexpr std::size_t kTasks = 64u;
  std::array<PipeCleanup, kTasks> pipes{};
  for (PipeCleanup &pipe : pipes) {
    pipe = MakePipe();
    TEST_ASSERT(pipe.read_fd >= 0 && pipe.write_fd >= 0);
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
                  .task_capacity = 80u,
                  .ready_queue_capacity = 80u,
                  .reactor_wait_capacity = 80u,
                  .observation_capacity = 128u,
                  .host_event_capacity = 128u,
              },
      },
      [&] {
        scoped = rund::task::scope([&] {
          for (std::size_t index = 0u; index < kTasks; ++index) {
            (void)rund::task::spawn(
                "reactor-registry-reader",
                rund::node::test_contract::reactor::AwaitReadable(
                    ready_fds[index].view(), &ready[index]));
          }
          (void)rund::task::spawn("reactor-registry-writer", [&] {
            const char byte = 'i';
            for (PipeCleanup &pipe : pipes) {
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
  TEST_ASSERT(stats.ready_expansion_scan_steps <= kTasks * 2u);
  TEST_ASSERT(stats.max_ready_batch >= kTasks);
  return 0;
}
