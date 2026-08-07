#include <unistd.h>

#include <array>
#include <rund/host.hpp>
#include <rund/session.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>
#include <vector>

#include "../../../../../../src/runtime/reactor/diagnostics.hpp"
#include "../../../../../../src/runtime/reactor/readiness/handle.hpp"
#include "../../../../../../src/runtime/task/scheduler/reactor/apply/policy.hpp"
#include "../../../../../../src/runtime/task/scheduler/reactor/backend.hpp"
#include "../../../../../../src/runtime/task/scheduler/reactor/change/queue.hpp"
#include "../../../../../../src/runtime/task/scheduler/reactor/model.hpp"
#include "../../../../../../src/runtime/task/scheduler/reactor/registry.hpp"
#include "../await.hpp"
#include "test/assert.hpp"

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

#if defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__)
void VerifyPartialApplyFailureIdentity() {
  using namespace rund::node;

  ReactorRuntime reactor{};
  ::rund::detail::task::StatStorage stats{};
  TEST_ASSERT(ReactorBackendApplyChanges(reactor, stats).disposition() ==
              ReactorApplyDisposition::Success);

  PipeCleanup registered = MakePipe();
  TEST_ASSERT(registered.read_fd >= 0 && registered.write_fd >= 0);
  const ReactorHandle registered_handle =
      ReactorHandleFromPublic(registered.read_fd);
  reactor.changes.push_back(ReactorRegistrationChange{
      .kind = ReactorRegistrationChange::Kind::Add,
      .handle = registered_handle,
      .interest = ReactorInterest::Read,
  });
  TEST_ASSERT(ReactorBackendApplyChanges(reactor, stats).disposition() ==
              ReactorApplyDisposition::Success);
  TEST_ASSERT(reactor.changes.empty());

  PipeCleanup invalid = MakePipe();
  TEST_ASSERT(invalid.read_fd >= 0 && invalid.write_fd >= 0);
  PipeCleanup suffix = MakePipe();
  TEST_ASSERT(suffix.read_fd >= 0 && suffix.write_fd >= 0);
  const ReactorHandle invalid_handle = ReactorHandleFromPublic(invalid.read_fd);
  const ReactorHandle suffix_handle = ReactorHandleFromPublic(suffix.read_fd);
  TEST_ASSERT(suffix_handle != invalid_handle);
  TEST_ASSERT(::close(invalid.read_fd) == 0);
  invalid.read_fd = -1;

  reactor.changes.push_back(ReactorRegistrationChange{
      .kind = ReactorRegistrationChange::Kind::Modify,
      .handle = registered_handle,
      .interest = ReactorInterest::Read,
  });
  reactor.changes.push_back(ReactorRegistrationChange{
      .kind = ReactorRegistrationChange::Kind::Add,
      .handle = invalid_handle,
      .interest = ReactorInterest::Read,
  });
  reactor.changes.push_back(ReactorRegistrationChange{
      .kind = ReactorRegistrationChange::Kind::Modify,
      .handle = invalid_handle,
      .interest = ReactorInterest::Write,
  });
  reactor.changes.push_back(ReactorRegistrationChange{
      .kind = ReactorRegistrationChange::Kind::Remove,
      .handle = invalid_handle,
      .best_effort = true,
  });
  reactor.changes.push_back(ReactorRegistrationChange{
      .kind = ReactorRegistrationChange::Kind::Add,
      .handle = suffix_handle,
      .interest = ReactorInterest::Read,
  });
  const ReactorApplyResult failed = ReactorChangeQueueApply(reactor, stats);
  TEST_ASSERT(failed.disposition() == ReactorApplyDisposition::Invalid);
  TEST_ASSERT(failed.invalid_change().handle() == invalid_handle);
  TEST_ASSERT(failed.invalid_change().fd_generation() == 0u);
  TEST_ASSERT(reactor.changes.size() == 4u);
  TEST_ASSERT(reactor.changes[0].handle == invalid_handle);
  TEST_ASSERT(reactor.changes[1].handle == invalid_handle);
  TEST_ASSERT(reactor.changes[2].handle == invalid_handle);
  TEST_ASSERT(reactor.changes[2].best_effort);
  TEST_ASSERT(reactor.changes[3].handle == suffix_handle);

  TEST_ASSERT(!ReactorChangeQueueAcknowledgeInvalid(
      reactor, ReactorInvalidChangeToken::observed(suffix_handle, 0u)));
  TEST_ASSERT(reactor.changes.size() == 4u);
  TEST_ASSERT(!ReactorChangeQueueAcknowledgeInvalid(
      reactor, ReactorInvalidChangeToken::observed(invalid_handle, 1u)));
  TEST_ASSERT(reactor.changes.size() == 4u);
  TEST_ASSERT(ReactorChangeQueueAcknowledgeInvalid(reactor,
                                                   failed.invalid_change()));
  TEST_ASSERT(reactor.changes.size() == 2u);
  TEST_ASSERT(reactor.changes[0].handle == invalid_handle);
  TEST_ASSERT(reactor.changes[0].best_effort);
  TEST_ASSERT(reactor.changes[1].handle == suffix_handle);
  TEST_ASSERT(ReactorBackendApplyChanges(reactor, stats).disposition() ==
              ReactorApplyDisposition::Success);
  TEST_ASSERT(reactor.changes.empty());

  TEST_ASSERT(ReactorRegistryPrepare(reactor, 1u));
  TEST_ASSERT(ReactorRegistryAddWait(
      reactor, ReactorWait{.task_id = 1u,
                           .wait_id = 1u,
                           .fd = suffix_handle,
                           .interest = ReactorInterest::Read}));
  PipeCleanup orphan = MakePipe();
  TEST_ASSERT(orphan.read_fd >= 0 && orphan.write_fd >= 0);
  const ReactorHandle orphan_handle = ReactorHandleFromPublic(orphan.read_fd);
  TEST_ASSERT(::close(orphan.read_fd) == 0);
  orphan.read_fd = -1;
  reactor.changes.push_back(ReactorRegistrationChange{
      .kind = ReactorRegistrationChange::Kind::Add,
      .handle = orphan_handle,
      .interest = ReactorInterest::Read,
  });
  reactor.changes.push_back(ReactorRegistrationChange{
      .kind = ReactorRegistrationChange::Kind::Modify,
      .handle = orphan_handle,
      .interest = ReactorInterest::Write,
  });
  reactor.changes.push_back(ReactorRegistrationChange{
      .kind = ReactorRegistrationChange::Kind::Modify,
      .handle = suffix_handle,
      .interest = ReactorInterest::Read,
  });
  TEST_ASSERT(ReactorBackendApplyChanges(reactor, stats).disposition() ==
              ReactorApplyDisposition::Success);
  TEST_ASSERT(reactor.changes.empty());
  TEST_ASSERT(ReactorRegistryFirstWait(reactor, suffix_handle) !=
              kNoReactorSlot);

  reactor.changes.push_back(ReactorRegistrationChange{
      .kind = ReactorRegistrationChange::Kind::Remove,
      .handle = registered_handle,
      .best_effort = true,
  });
  TEST_ASSERT(ReactorBackendApplyChanges(reactor, stats).disposition() ==
              ReactorApplyDisposition::Success);
  ReactorCloseRuntime(reactor);
}
#endif

} // namespace

int RunRuntimeTaskReactorBatchApplyContract() {
#if defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__)
  VerifyPartialApplyFailureIdentity();
#endif

  rund::node::ReactorRuntime policy_reactor{};
  policy_reactor.changes.push_back(rund::node::ReactorRegistrationChange{
      .kind = rund::node::ReactorRegistrationChange::Kind::Add,
      .handle = rund::node::ReactorHandleFromPublic(3),
      .interest = rund::node::ReactorInterest::Read});
  {
    rund::node::ReactorApplyBatchScope scope{policy_reactor};
    TEST_ASSERT(rund::node::ReactorApplyPolicyShouldDefer(
        policy_reactor, /*ready_depth=*/0u, /*force=*/false));
    TEST_ASSERT(!rund::node::ReactorApplyPolicyShouldDefer(
        policy_reactor, /*ready_depth=*/0u, /*force=*/true));
  }
  TEST_ASSERT(!policy_reactor.apply_policy.defer_registration_apply);
  TEST_ASSERT(policy_reactor.apply_policy.defer_depth == 0u);
  TEST_ASSERT(policy_reactor.apply_policy.batch_add_defer_depth == 0u);

  constexpr std::size_t kTasks = 16u;
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
                  .task_capacity = 32u,
                  .ready_queue_capacity = 32u,
                  .reactor_wait_capacity = 32u,
                  .observation_capacity = 64u,
                  .host_event_capacity = 64u,
              },
      },
      [&] {
        scoped = rund::task::scope([&] {
          for (std::size_t index = 0u; index < kTasks; ++index) {
            (void)rund::task::spawn(
                "reactor-batch-apply-reader",
                rund::node::test_contract::reactor::AwaitReadable(
                    ready_fds[index].view(), &ready[index]));
          }
          (void)rund::task::spawn("reactor-batch-apply-writer", [&] {
            const char byte = 'b';
            for (PipeCleanup &pipe : pipes) {
              const ssize_t written = ::write(pipe.write_fd, &byte, 1u);
              if (written != 1) {
                writes_ok = false;
              }
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
  TEST_ASSERT(stats.max_registered_fds == kTasks);
  TEST_ASSERT(stats.remove_calls <= kTasks);
  TEST_ASSERT(stats.deferred_remove_marks == kTasks);
  TEST_ASSERT(stats.registration_apply_calls <= 2u);
  TEST_ASSERT(stats.max_registration_changes_per_apply >= kTasks);
  return 0;
}
