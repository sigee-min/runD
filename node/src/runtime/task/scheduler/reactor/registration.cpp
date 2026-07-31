#include "registration.hpp"

#include "registry.hpp"

#include <algorithm>

#include "../../../reactor/diagnostics.hpp"

namespace rund::node {
namespace {

[[nodiscard]] bool ReserveChanges(const ReactorRuntime &reactor,
                                  const std::size_t count) noexcept {
  return count <= reactor.changes.capacity() - reactor.changes.size();
}

[[nodiscard]] bool PushChange(
    ReactorRuntime &reactor, const ReactorRegistrationChange::Kind kind,
    const ReactorHandle fd, const ReactorInterest interest,
    const std::uint64_t fd_generation,
    const bool best_effort = false) noexcept {
  if (!ReserveChanges(reactor, 1u)) {
    return false;
  }
  reactor.changes.push_back(ReactorRegistrationChange{
      .kind = kind,
      .handle = fd,
      .interest = interest,
      .fd_generation = fd_generation,
      .best_effort = best_effort,
  });
  return true;
}

void SetDeferred(ReactorRuntime &reactor, ReactorFdState &state,
                 const bool deferred) noexcept {
  if (state.remove_deferred == deferred) {
    return;
  }
  state.remove_deferred = deferred;
  if (deferred) {
    ++reactor.registry.deferred_removes;
  } else {
    --reactor.registry.deferred_removes;
  }
}

void RememberDeferredStats(const ReactorRuntime &reactor) noexcept {
  RecordReactorDeferredRemovePending(reactor.registry.deferred_removes);
}

void ReleaseRawFd(ReactorFdState &state) noexcept {
  if (state.identity_guard != kInvalidReactorHandle) {
    ReleaseReactorPlatformHandle(state.identity_guard);
    state.identity_guard = kInvalidReactorHandle;
  }
}

[[nodiscard]] bool
SameRawFd(const ReactorFdState &state,
          const ReactorPlatformHandleIdentity identity) noexcept {
  return state.fd_identity_valid && identity.valid &&
         state.fd_device == identity.device &&
         state.fd_inode == identity.inode && state.fd_mode == identity.mode;
}

void RememberRawFd(ReactorFdState &state,
                   const ReactorPlatformHandleIdentity identity,
                   const ReactorHandle identity_guard) noexcept {
  ReleaseRawFd(state);
  state.fd_device = identity.device;
  state.fd_inode = identity.inode;
  state.fd_mode = identity.mode;
  state.identity_guard = identity_guard;
  state.fd_identity_valid = identity.valid;
}

[[nodiscard]] bool Empty(const ReactorFdState &state) noexcept {
  return state.wait_count == 0u &&
         state.backend_interest == ReactorInterest::None && !state.registered &&
         !state.remove_deferred;
}

} // namespace

bool ReactorRegistrationCollectForWaitAdd(
    ReactorRuntime &reactor, const ReactorHandle fd,
    const ReactorInterest current_interest,
    const std::uint64_t fd_generation) noexcept {
  ReactorFdState *state = ReactorRegistryFindFd(reactor, fd);
  if (state == nullptr || state->wait_count == 0u ||
      current_interest == ReactorInterest::None) {
    return false;
  }
  const ReactorPlatformHandleIdentity raw_identity =
      fd_generation == 0u ? DescribeReactorPlatformHandle(fd)
                          : ReactorPlatformHandleIdentity{};
  if (fd_generation == 0u && !raw_identity.valid) {
    return false;
  }
  if (!ReserveChanges(reactor, 1u)) {
    return false;
  }

  const bool raw_replaced =
      fd_generation == 0u && !SameRawFd(*state, raw_identity);
  if (raw_replaced) {
    const ReactorHandle identity_guard = RetainReactorPlatformHandle(fd);
    if (identity_guard == kInvalidReactorHandle) {
      return false;
    }
    const ReactorRegistrationChange::Kind kind =
        state->registered ? ReactorRegistrationChange::Kind::Modify
                          : ReactorRegistrationChange::Kind::Add;
    state->fd_generation = 0u;
    SetDeferred(reactor, *state, false);
    RememberRawFd(*state, raw_identity, identity_guard);
    if (!PushChange(reactor, kind, fd, current_interest, 0u)) {
      return false;
    }
    state->registered = true;
    state->backend_interest = current_interest;
    return true;
  }
  if (state->fd_generation != fd_generation) {
    state->backend_interest = ReactorInterest::None;
    state->fd_generation = fd_generation;
    state->registered = false;
    SetDeferred(reactor, *state, false);
    RememberRawFd(*state, raw_identity, kInvalidReactorHandle);
  }
  if (state->remove_deferred && state->registered &&
      state->backend_interest == current_interest) {
    SetDeferred(reactor, *state, false);
    RecordReactorDeferredRemoveCancellation();
    return true;
  }
  SetDeferred(reactor, *state, false);
  if (!state->registered ||
      state->backend_interest == ReactorInterest::None) {
    if (!PushChange(reactor, ReactorRegistrationChange::Kind::Add, fd,
                    current_interest, fd_generation)) {
      return false;
    }
  } else if (state->backend_interest != current_interest) {
    if (!PushChange(reactor, ReactorRegistrationChange::Kind::Modify, fd,
                    current_interest, fd_generation)) {
      return false;
    }
  }
  state->registered = true;
  state->backend_interest = current_interest;
  return true;
}

bool ReactorRegistrationCollectForWaitRemove(
    ReactorRuntime &reactor, const ReactorHandle fd,
    const ReactorInterest previous_interest,
    const ReactorInterest current_interest) noexcept {
  ReactorFdState *state = ReactorRegistryFindFd(reactor, fd);
  if (state == nullptr) {
    return true;
  }
  if (current_interest == previous_interest) {
    return true;
  }
  if (current_interest == ReactorInterest::None) {
    if (state->registered) {
      const bool newly_deferred = !state->remove_deferred;
      SetDeferred(reactor, *state, true);
      if (newly_deferred) {
        RecordReactorDeferredRemoveMark();
      }
      RememberDeferredStats(reactor);
    } else {
      SetDeferred(reactor, *state, false);
      ReleaseRawFd(*state);
      if (Empty(*state)) {
        static_cast<void>(ReactorRegistryEraseFd(reactor, fd));
      }
    }
    return true;
  }
  if (!ReserveChanges(reactor, 1u)) {
    return false;
  }
  SetDeferred(reactor, *state, false);
  if (state->backend_interest == ReactorInterest::None) {
    if (!PushChange(reactor, ReactorRegistrationChange::Kind::Add, fd,
                    current_interest, state->fd_generation)) {
      return false;
    }
  } else if (state->backend_interest != current_interest) {
    if (!PushChange(reactor, ReactorRegistrationChange::Kind::Modify, fd,
                    current_interest, state->fd_generation)) {
      return false;
    }
  }
  state->registered = true;
  state->backend_interest = current_interest;
  return true;
}

bool ReactorRegistrationFlushDeferredRemoves(
    ReactorRuntime &reactor) noexcept {
  const std::size_t pending = reactor.registry.deferred_removes;
  if (pending == 0u) {
    return true;
  }
  if (!ReserveChanges(reactor, pending)) {
    return false;
  }
  for (ReactorFdState &state : reactor.registry.fds) {
    if (!state.remove_deferred) {
      continue;
    }
    if (state.registered &&
        !PushChange(reactor, ReactorRegistrationChange::Kind::Remove, state.fd,
                    ReactorInterest::None, state.fd_generation, true)) {
      return false;
    }
    state.backend_interest = ReactorInterest::None;
    state.registered = false;
    SetDeferred(reactor, state, false);
    ReleaseRawFd(state);
    RecordReactorDeferredRemoveFlush();
  }
  reactor.registry.fds.erase(
      std::remove_if(reactor.registry.fds.begin(), reactor.registry.fds.end(),
                     [](const ReactorFdState &state) { return Empty(state); }),
      reactor.registry.fds.end());
  return true;
}

bool ReactorRegistrationHasDeferredRemoves(
    const ReactorRuntime &reactor) noexcept {
  return reactor.registry.deferred_removes != 0u;
}

std::size_t ReactorRegistrationDeferredRemoveCount(
    const ReactorRuntime &reactor) noexcept {
  return reactor.registry.deferred_removes;
}

void ReactorRegistrationForgetGeneration(
    ReactorRuntime &reactor, const ReactorHandle fd,
    const std::uint64_t fd_generation) noexcept {
  reactor.changes.erase(
      std::remove_if(
          reactor.changes.begin(), reactor.changes.end(),
          [fd, fd_generation](const ReactorRegistrationChange &change) {
            return change.handle == fd &&
                   change.fd_generation == fd_generation;
          }),
      reactor.changes.end());
  ReactorFdState *const state = ReactorRegistryFindFd(reactor, fd);
  if (state == nullptr || state->fd_generation != fd_generation) {
    return;
  }
  SetDeferred(reactor, *state, false);
  ReleaseRawFd(*state);
  state->backend_interest = ReactorInterest::None;
  state->registered = false;
  state->fd_generation = 0u;
  state->fd_identity_valid = false;
  if (state->wait_count == 0u) {
    static_cast<void>(ReactorRegistryEraseFd(reactor, fd));
  }
}

void ReactorRegistrationClear(ReactorRuntime &reactor) noexcept {
  for (ReactorFdState &state : reactor.registry.fds) {
    ReleaseRawFd(state);
  }
  ReactorRegistryClear(reactor);
}

} // namespace rund::node
