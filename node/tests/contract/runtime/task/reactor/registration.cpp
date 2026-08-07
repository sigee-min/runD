#include "test/assert.hpp"

#include <rund/host.hpp>
#include <rund/session.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include "../../../../../src/runtime/reactor/diagnostics.hpp"
#include "../../../../../src/runtime/reactor/readiness/handle.hpp"
#include "../../../../../src/runtime/reactor/readiness/mask.hpp"
#include "../../../../../src/runtime/task/scheduler/reactor/registration.hpp"
#include "../../../../../src/runtime/task/scheduler/reactor/registry.hpp"

#include <cerrno>
#include <type_traits>
#include <vector>

#include <fcntl.h>
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

[[nodiscard]] int GetFdFlags(const int fd) noexcept {
  int result = -1;
  do {
    errno = 0;
    result = ::fcntl(fd, F_GETFD);
  } while (result == -1 && errno == EINTR);
  return result;
}

void VerifyFdRegistrationState() {
  using rund::node::ReactorFdRegistration;
  using rund::node::ReactorFdRegistrationPhase;
  using rund::node::ReactorHandleFromPublic;
  using rund::node::ReactorInterest;
  using rund::node::ReactorRegistrationChange;
  using rund::node::ReactorWait;

  static_assert(!std::is_aggregate_v<ReactorFdRegistration>);
  static_assert(std::is_trivially_copyable_v<ReactorFdRegistration>);
  constexpr ReactorFdRegistration idle = ReactorFdRegistration::idle();
  constexpr ReactorFdRegistration active =
      ReactorFdRegistration::active(ReactorInterest::Read);
  constexpr ReactorFdRegistration deferred =
      ReactorFdRegistration::deferred_remove(ReactorInterest::Read |
                                             ReactorInterest::Write);
  static_assert(idle.is_idle());
  static_assert(idle.interest() == ReactorInterest::None);
  static_assert(active.phase() == ReactorFdRegistrationPhase::Active);
  static_assert(active.interest() == ReactorInterest::Read);
  static_assert(deferred.phase() ==
                ReactorFdRegistrationPhase::DeferredRemove);
  static_assert(deferred.interest() ==
                (ReactorInterest::Read | ReactorInterest::Write));
  static_assert(ReactorFdRegistration::active(ReactorInterest::None)
                    .is_idle());
  static_assert(
      ReactorFdRegistration::deferred_remove(ReactorInterest::None)
          .is_idle());

  rund::node::ReactorRuntime reactor{};
  reactor.changes.reserve(2u);
  TEST_ASSERT(rund::node::ReactorRegistryPrepare(reactor, 2u));
  constexpr std::uint64_t kGeneration = 7u;
  const rund::node::ReactorHandle fd = ReactorHandleFromPublic(17);
  const ReactorWait wait{.task_id = 3u,
                         .wait_id = 5u,
                         .fd_generation = kGeneration,
                         .fd = fd,
                         .interest = ReactorInterest::Read};
  TEST_ASSERT(rund::node::ReactorRegistryAddWait(reactor, wait));
  const rund::node::ReactorFdState *state =
      rund::node::ReactorRegistryFindFd(reactor, fd);
  TEST_ASSERT(state != nullptr);
  TEST_ASSERT(state->wait_count == 1u);
  TEST_ASSERT(state->registration.is_idle());

  TEST_ASSERT(
      rund::node::ReactorRegistryCollectChangesForWaitAdd(reactor, wait));
  state = rund::node::ReactorRegistryFindFd(reactor, fd);
  TEST_ASSERT(state != nullptr);
  TEST_ASSERT(state->registration.phase() ==
              ReactorFdRegistrationPhase::Active);
  TEST_ASSERT(state->registration.interest() == ReactorInterest::Read);
  TEST_ASSERT(reactor.registry.deferred_removes == 0u);
  TEST_ASSERT(reactor.changes.size() == 1u);
  TEST_ASSERT(reactor.changes[0].kind() ==
              ReactorRegistrationChange::Kind::Add);

  rund::node::ReactorRegistrationForgetGeneration(reactor, fd,
                                                   kGeneration + 1u);
  state = rund::node::ReactorRegistryFindFd(reactor, fd);
  TEST_ASSERT(state != nullptr);
  TEST_ASSERT(state->registration.phase() ==
              ReactorFdRegistrationPhase::Active);
  TEST_ASSERT(reactor.changes.size() == 1u);

  rund::node::ReactorRegistrationForgetGeneration(reactor, fd, kGeneration);
  state = rund::node::ReactorRegistryFindFd(reactor, fd);
  TEST_ASSERT(state != nullptr);
  TEST_ASSERT(state->wait_count == 1u);
  TEST_ASSERT(state->registration.is_idle());
  TEST_ASSERT(state->fd_generation == 0u);
  TEST_ASSERT(reactor.changes.empty());
  TEST_ASSERT(
      rund::node::ReactorRegistryCollectChangesForWaitAdd(reactor, wait));
  state = rund::node::ReactorRegistryFindFd(reactor, fd);
  TEST_ASSERT(state != nullptr);
  TEST_ASSERT(state->registration.phase() ==
              ReactorFdRegistrationPhase::Active);
  TEST_ASSERT(reactor.changes.size() == 1u);

  ReactorWait removed{};
  ReactorInterest previous = ReactorInterest::None;
  TEST_ASSERT(rund::node::ReactorRegistryRemoveWait(
      reactor, wait.wait_id, &removed, &previous));
  state = rund::node::ReactorRegistryFindFd(reactor, fd);
  TEST_ASSERT(state != nullptr);
  TEST_ASSERT(state->wait_count == 0u);
  TEST_ASSERT(state->registration.phase() ==
              ReactorFdRegistrationPhase::Active);

  TEST_ASSERT(rund::node::ReactorRegistryCollectChangesForWaitRemove(
      reactor, fd, previous));
  state = rund::node::ReactorRegistryFindFd(reactor, fd);
  TEST_ASSERT(state != nullptr);
  TEST_ASSERT(state->registration.phase() ==
              ReactorFdRegistrationPhase::DeferredRemove);
  TEST_ASSERT(state->registration.interest() == ReactorInterest::Read);
  TEST_ASSERT(reactor.registry.deferred_removes == 1u);

  TEST_ASSERT(rund::node::ReactorRegistryAddWait(reactor, wait));
  state = rund::node::ReactorRegistryFindFd(reactor, fd);
  TEST_ASSERT(state != nullptr);
  TEST_ASSERT(state->wait_count == 1u);
  TEST_ASSERT(state->registration.phase() ==
              ReactorFdRegistrationPhase::DeferredRemove);

  TEST_ASSERT(
      rund::node::ReactorRegistryCollectChangesForWaitAdd(reactor, wait));
  state = rund::node::ReactorRegistryFindFd(reactor, fd);
  TEST_ASSERT(state != nullptr);
  TEST_ASSERT(state->registration.phase() ==
              ReactorFdRegistrationPhase::Active);
  TEST_ASSERT(reactor.registry.deferred_removes == 0u);
  TEST_ASSERT(reactor.changes.size() == 1u);

  TEST_ASSERT(rund::node::ReactorRegistryRemoveWait(
      reactor, wait.wait_id, &removed, &previous));
  TEST_ASSERT(rund::node::ReactorRegistryCollectChangesForWaitRemove(
      reactor, fd, previous));
  TEST_ASSERT(reactor.registry.deferred_removes == 1u);
  TEST_ASSERT(rund::node::ReactorRegistrationFlushDeferredRemoves(reactor));
  TEST_ASSERT(reactor.registry.deferred_removes == 0u);
  TEST_ASSERT(rund::node::ReactorRegistryFindFd(reactor, fd) == nullptr);
  TEST_ASSERT(reactor.changes.size() == 2u);
  TEST_ASSERT(reactor.changes[1].kind() ==
              ReactorRegistrationChange::Kind::CleanupRemove);
  TEST_ASSERT(reactor.changes[1].fd_generation() == kGeneration);
}

void VerifyRawFdRegistrationIdentity() {
  using rund::node::ReactorFdRegistrationPhase;
  using rund::node::ReactorHandleForPublic;
  using rund::node::ReactorHandleFromPublic;
  using rund::node::ReactorInterest;
  using rund::node::ReactorPlatformHandleIdentityDisposition;
  using rund::node::ReactorRegistrationChange;
  using rund::node::ReactorWait;

  PipeCleanup pipe = MakePipe();
  TEST_ASSERT(pipe.read_fd >= 0 && pipe.write_fd >= 0);
  rund::node::ReactorRuntime reactor{};
  reactor.changes.reserve(3u);
  TEST_ASSERT(rund::node::ReactorRegistryPrepare(reactor, 1u));
  const rund::node::ReactorHandle fd =
      ReactorHandleFromPublic(pipe.read_fd);
  const ReactorWait wait{.task_id = 4u,
                         .wait_id = 6u,
                         .fd = fd,
                         .interest = ReactorInterest::Read};
  TEST_ASSERT(rund::node::ReactorRegistryAddWait(reactor, wait));
  TEST_ASSERT(
      rund::node::ReactorRegistryCollectChangesForWaitAdd(reactor, wait));
  const rund::node::ReactorFdState *state =
      rund::node::ReactorRegistryFindFd(reactor, fd);
  TEST_ASSERT(state != nullptr);
  TEST_ASSERT(state->registration.phase() ==
              ReactorFdRegistrationPhase::Active);
  TEST_ASSERT(state->fd_identity.disposition() ==
              ReactorPlatformHandleIdentityDisposition::Described);
  TEST_ASSERT(state->identity_guard !=
              rund::node::kInvalidReactorHandle);
  const int identity_guard = ReactorHandleForPublic(state->identity_guard);
  TEST_ASSERT(identity_guard >= 0);
  TEST_ASSERT(GetFdFlags(identity_guard) >= 0);

  ReactorWait removed{};
  ReactorInterest previous = ReactorInterest::None;
  TEST_ASSERT(rund::node::ReactorRegistryRemoveWait(
      reactor, wait.wait_id, &removed, &previous));
  TEST_ASSERT(rund::node::ReactorRegistryCollectChangesForWaitRemove(
      reactor, fd, previous));
  state = rund::node::ReactorRegistryFindFd(reactor, fd);
  TEST_ASSERT(state != nullptr);
  TEST_ASSERT(state->registration.phase() ==
              ReactorFdRegistrationPhase::DeferredRemove);
  TEST_ASSERT(ReactorHandleForPublic(state->identity_guard) == identity_guard);
  TEST_ASSERT(GetFdFlags(identity_guard) >= 0);

  PipeCleanup replacement = MakePipe();
  TEST_ASSERT(replacement.read_fd >= 0 && replacement.write_fd >= 0);
  const int reused_fd = pipe.read_fd;
  TEST_ASSERT(::close(pipe.read_fd) == 0);
  pipe.read_fd = -1;
  TEST_ASSERT(::dup2(replacement.read_fd, reused_fd) == reused_fd);
  pipe.read_fd = reused_fd;

  TEST_ASSERT(rund::node::ReactorRegistryAddWait(reactor, wait));
  TEST_ASSERT(
      rund::node::ReactorRegistryCollectChangesForWaitAdd(reactor, wait));
  state = rund::node::ReactorRegistryFindFd(reactor, fd);
  TEST_ASSERT(state != nullptr);
  TEST_ASSERT(state->registration.phase() ==
              ReactorFdRegistrationPhase::Active);
  TEST_ASSERT(state->registration.interest() == ReactorInterest::Read);
  TEST_ASSERT(reactor.registry.deferred_removes == 0u);
  TEST_ASSERT(reactor.changes.size() == 2u);
  TEST_ASSERT(reactor.changes[1].kind() ==
              ReactorRegistrationChange::Kind::Modify);
  TEST_ASSERT(GetFdFlags(identity_guard) == -1 && errno == EBADF);
  const int replacement_guard =
      ReactorHandleForPublic(state->identity_guard);
  TEST_ASSERT(replacement_guard >= 0 && replacement_guard != identity_guard);
  TEST_ASSERT(GetFdFlags(replacement_guard) >= 0);

  TEST_ASSERT(rund::node::ReactorRegistryRemoveWait(
      reactor, wait.wait_id, &removed, &previous));
  TEST_ASSERT(rund::node::ReactorRegistryCollectChangesForWaitRemove(
      reactor, fd, previous));
  state = rund::node::ReactorRegistryFindFd(reactor, fd);
  TEST_ASSERT(state != nullptr);
  TEST_ASSERT(state->registration.phase() ==
              ReactorFdRegistrationPhase::DeferredRemove);
  TEST_ASSERT(reactor.registry.deferred_removes == 1u);

  TEST_ASSERT(rund::node::ReactorRegistrationFlushDeferredRemoves(reactor));
  TEST_ASSERT(rund::node::ReactorRegistryFindFd(reactor, fd) == nullptr);
  TEST_ASSERT(GetFdFlags(replacement_guard) == -1 && errno == EBADF);
}

} // namespace

int RunRuntimeTaskReactorRegistrationContract() {
  VerifyFdRegistrationState();
  VerifyRawFdRegistrationIdentity();
  PipeCleanup pipe = MakePipe();
  TEST_ASSERT(pipe.read_fd >= 0 && pipe.write_fd >= 0);
  rund::host::io::Fd ready_fd =
      rund::host::io::take_native_fd(::dup(pipe.read_fd));
  TEST_ASSERT(ready_fd);

  rund::node::ResetReactorBackendStats();
  std::vector<int> wake_order{};
  rund::task::IoResult first_ready{};
  rund::task::IoResult second_ready{};
  rund::task::Status joined{};
  bool write_ok = false;

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
        auto wait_first = [&]() -> rund::task::Task<void> {
          first_ready = co_await rund::host::io::readable(ready_fd.view());
          if (first_ready.ok()) {
            wake_order.push_back(1);
          }
        };
        const rund::task::Handle first =
            rund::task::spawn("reactor-registration-first", wait_first());
        auto wait_second = [&]() -> rund::task::Task<void> {
          second_ready = co_await rund::host::io::readable(ready_fd.view());
          if (second_ready.ok()) {
            wake_order.push_back(2);
          }
        };
        const rund::task::Handle second =
            rund::task::spawn("reactor-registration-second", wait_second());
        const rund::task::Handle writer =
            rund::task::spawn("reactor-registration-writer", [&] {
              const char byte = 'r';
              write_ok = ::write(pipe.write_fd, &byte, 1u) == 1;
            });
        joined = rund::task::join(first, second, writer);
      });

  const rund::node::ReactorBackendStats stats =
      rund::node::ReactorBackendStatsSnapshot();

  TEST_ASSERT(report.ok());
  TEST_ASSERT(joined.ok());
  TEST_ASSERT(write_ok);
  TEST_ASSERT(first_ready.ok());
  TEST_ASSERT(second_ready.ok());
  TEST_ASSERT(wake_order.size() == 2u);
  TEST_ASSERT(wake_order[0] == 1);
  TEST_ASSERT(wake_order[1] == 2);
  TEST_ASSERT(stats.add_calls == 1u);
  TEST_ASSERT(stats.modify_calls == 0u);
  TEST_ASSERT(stats.remove_calls <= 1u);
  TEST_ASSERT(stats.deferred_remove_marks == 1u);
  TEST_ASSERT(stats.max_registered_fds == 1u);
  return 0;
}
