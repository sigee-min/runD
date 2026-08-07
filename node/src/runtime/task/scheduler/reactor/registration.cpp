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

[[nodiscard]] bool
PushChange(ReactorRuntime &reactor,
           const ReactorRegistrationChange change) noexcept {
  if (!ReserveChanges(reactor, 1u)) {
    return false;
  }
  reactor.changes.push_back(change);
  return true;
}

void SetRegistration(ReactorRuntime &reactor, ReactorFdState &state,
                     const ReactorFdRegistration registration) noexcept {
  const bool was_deferred =
      state.registration.phase() ==
      ReactorFdRegistrationPhase::DeferredRemove;
  const bool will_be_deferred =
      registration.phase() == ReactorFdRegistrationPhase::DeferredRemove;
  if (!was_deferred && will_be_deferred) {
    ++reactor.registry.deferred_removes;
  } else if (was_deferred && !will_be_deferred &&
             reactor.registry.deferred_removes != 0u) {
    --reactor.registry.deferred_removes;
  }
  state.registration = registration;
}

void RememberDeferredStats(const ReactorRuntime &reactor) noexcept {
  RecordReactorDeferredRemovePending(reactor.registry.deferred_removes);
}

void ForgetRawFd(ReactorFdState &state) noexcept {
  if (state.identity_guard != kInvalidReactorHandle) {
    ReleaseReactorPlatformHandle(state.identity_guard);
    state.identity_guard = kInvalidReactorHandle;
  }
  state.fd_identity = ReactorPlatformHandleIdentity::invalid();
}

void ForgetRegistration(ReactorRuntime &reactor,
                        ReactorFdState &state) noexcept {
  SetRegistration(reactor, state, ReactorFdRegistration::idle());
  ForgetRawFd(state);
}

[[nodiscard]] bool
SameRawFd(const ReactorFdState &state,
          const ReactorPlatformHandleIdentity identity) noexcept {
  return state.fd_identity.same_object(identity);
}

void RememberRawFd(ReactorFdState &state,
                   const ReactorPlatformHandleIdentity identity,
                   const ReactorHandle identity_guard) noexcept {
  ForgetRawFd(state);
  state.fd_identity = identity;
  state.identity_guard = identity_guard;
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
                          : ReactorPlatformHandleIdentity::invalid();
  if (fd_generation == 0u &&
      raw_identity.disposition() !=
          ReactorPlatformHandleIdentityDisposition::Described) {
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
    const ReactorRegistrationChange change =
        !state->registration.is_idle()
            ? ReactorRegistrationChange::modify(fd, current_interest, 0u)
            : ReactorRegistrationChange::add(fd, current_interest, 0u);
    state->fd_generation = 0u;
    RememberRawFd(*state, raw_identity, identity_guard);
    if (!PushChange(reactor, change)) {
      return false;
    }
    SetRegistration(reactor, *state,
                    ReactorFdRegistration::active(current_interest));
    return true;
  }
  if (state->fd_generation != fd_generation) {
    ForgetRegistration(reactor, *state);
    state->fd_generation = fd_generation;
  }
  if (state->registration.phase() ==
          ReactorFdRegistrationPhase::DeferredRemove &&
      state->registration.interest() == current_interest) {
    SetRegistration(reactor, *state,
                    ReactorFdRegistration::active(current_interest));
    RecordReactorDeferredRemoveCancellation();
    return true;
  }
  if (state->registration.is_idle()) {
    if (!PushChange(reactor, ReactorRegistrationChange::add(
                                 fd, current_interest, fd_generation))) {
      return false;
    }
  } else if (state->registration.interest() != current_interest) {
    if (!PushChange(reactor, ReactorRegistrationChange::modify(
                                 fd, current_interest, fd_generation))) {
      return false;
    }
  }
  SetRegistration(reactor, *state,
                  ReactorFdRegistration::active(current_interest));
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
    if (!state->registration.is_idle()) {
      const bool newly_deferred =
          state->registration.phase() !=
          ReactorFdRegistrationPhase::DeferredRemove;
      SetRegistration(
          reactor, *state,
          ReactorFdRegistration::deferred_remove(
              state->registration.interest()));
      if (newly_deferred) {
        RecordReactorDeferredRemoveMark();
      }
      RememberDeferredStats(reactor);
    } else {
      ForgetRegistration(reactor, *state);
      if (state->erasable()) {
        static_cast<void>(ReactorRegistryEraseFd(reactor, fd));
      }
    }
    return true;
  }
  if (!ReserveChanges(reactor, 1u)) {
    return false;
  }
  if (state->registration.is_idle()) {
    if (!PushChange(reactor, ReactorRegistrationChange::add(
                                 fd, current_interest,
                                 state->fd_generation))) {
      return false;
    }
  } else if (state->registration.interest() != current_interest) {
    if (!PushChange(reactor, ReactorRegistrationChange::modify(
                                 fd, current_interest,
                                 state->fd_generation))) {
      return false;
    }
  }
  SetRegistration(reactor, *state,
                  ReactorFdRegistration::active(current_interest));
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
    if (state.registration.phase() !=
        ReactorFdRegistrationPhase::DeferredRemove) {
      continue;
    }
    if (!PushChange(reactor, ReactorRegistrationChange::cleanup_remove(
                                 state.fd, state.fd_generation))) {
      return false;
    }
    ForgetRegistration(reactor, state);
    RecordReactorDeferredRemoveFlush();
  }
  reactor.registry.fds.erase(
      std::remove_if(reactor.registry.fds.begin(), reactor.registry.fds.end(),
                     [](const ReactorFdState &state) {
                       return state.erasable();
                     }),
      reactor.registry.fds.end());
  return true;
}

bool ReactorRegistrationHasDeferredRemoves(
    const ReactorRuntime &reactor) noexcept {
  return reactor.registry.deferred_removes != 0u;
}

void ReactorRegistrationForgetGeneration(
    ReactorRuntime &reactor, const ReactorHandle fd,
    const std::uint64_t fd_generation) noexcept {
  reactor.changes.erase(
      std::remove_if(
          reactor.changes.begin(), reactor.changes.end(),
          [fd, fd_generation](const ReactorRegistrationChange &change) {
            return change.handle() == fd &&
                   change.fd_generation() == fd_generation;
          }),
      reactor.changes.end());
  ReactorFdState *const state = ReactorRegistryFindFd(reactor, fd);
  if (state == nullptr || state->fd_generation != fd_generation) {
    return;
  }
  ForgetRegistration(reactor, *state);
  state->fd_generation = 0u;
  if (state->wait_count == 0u) {
    static_cast<void>(ReactorRegistryEraseFd(reactor, fd));
  }
}

void ReactorRegistrationClear(ReactorRuntime &reactor) noexcept {
  for (ReactorFdState &state : reactor.registry.fds) {
    ForgetRawFd(state);
  }
  ReactorRegistryClear(reactor);
}

} // namespace rund::node
