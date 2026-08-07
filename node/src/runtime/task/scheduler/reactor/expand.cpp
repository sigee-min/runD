#include "expand.hpp"

#include <limits>

#include "../../../reactor/diagnostics.hpp"
#include "../../../reactor/readiness/mask.hpp"
#include "registry.hpp"

namespace rund::node {
namespace {

[[nodiscard]] bool ReserveReady(ReactorRuntime &reactor,
                                const std::size_t capacity) noexcept {
  try {
    reactor.ready.clear();
    reactor.ready.reserve(capacity);
  } catch (...) {
    reactor.ready.clear();
    return false;
  }
  return true;
}

void PushReady(ReactorRuntime &reactor, const std::uint64_t wait_id,
               const std::uint64_t task_id, const ReactorHandle fd,
               const ReactorInterest interest, const ReactorEvent events,
               const ReactorReadyDisposition disposition) {
  reactor.ready.push_back(ReactorReady{
      .wait_id = wait_id,
      .task_id = task_id,
      .fd = fd,
      .interest = interest,
      .events = events,
      .disposition = disposition,
  });
}

void PushWaitReady(ReactorRuntime &reactor, const ReactorWait &wait,
                   const ReactorEvent events,
                   const ReactorReadyDisposition disposition) {
  PushReady(reactor, wait.wait_id, wait.task_id, wait.fd, wait.interest, events,
            disposition);
}

} // namespace

bool ReactorExpandPlatformReady(
    ReactorRuntime &reactor,
    const std::span<const ReactorPlatformReady> ready) noexcept {
  std::size_t capacity = 0u;
  for (const ReactorPlatformReady &platform_ready : ready) {
    const ReactorFdState *const fd =
        ReactorRegistryFindFd(reactor, platform_ready.handle);
    if (fd == nullptr) {
      continue;
    }
    if (std::numeric_limits<std::size_t>::max() - capacity < fd->wait_count) {
      return false;
    }
    capacity += fd->wait_count;
  }
  if (!ReserveReady(reactor, capacity)) {
    return false;
  }
  for (const ReactorPlatformReady &platform_ready : ready) {
    const ReactorFdState *const fd =
        ReactorRegistryFindFd(reactor, platform_ready.handle);
    if (fd == nullptr) {
      continue;
    }
    RecordReactorReadyExpansionScanStep(fd->wait_count);
    for (std::uint32_t slot =
             ReactorRegistryFirstWait(reactor, platform_ready.handle);
         slot != kNoReactorSlot; slot = ReactorRegistryNextWait(reactor, slot)) {
      const ReactorWait *const wait = ReactorRegistrySlotWait(reactor, slot);
      if (wait == nullptr) {
        return false;
      }
      if (!platform_ready.invalid &&
          !ReactorEventsMatch(platform_ready.events, wait->interest)) {
        continue;
      }
      PushWaitReady(reactor, *wait, platform_ready.events,
                    platform_ready.invalid ? ReactorReadyDisposition::Invalid
                                           : ReactorReadyDisposition::Ready);
    }
  }
  return true;
}

bool ReactorExpandInvalidHandle(ReactorRuntime &reactor,
                                const ReactorHandle handle) noexcept {
  const ReactorFdState *const fd = ReactorRegistryFindFd(reactor, handle);
  if (!ReserveReady(reactor, fd == nullptr ? 0u : fd->wait_count)) {
    return false;
  }
  for (std::uint32_t slot = ReactorRegistryFirstWait(reactor, handle);
       slot != kNoReactorSlot; slot = ReactorRegistryNextWait(reactor, slot)) {
    const ReactorWait *const wait = ReactorRegistrySlotWait(reactor, slot);
    if (wait == nullptr) {
      return false;
    }
    PushWaitReady(reactor, *wait, ReactorEventsForInterest(wait->interest),
                  ReactorReadyDisposition::Invalid);
  }
  return true;
}

bool ReactorExpandPollFailure(ReactorRuntime &reactor) noexcept {
  const std::size_t count = ReactorRegistrySize(reactor);
  if (!ReserveReady(reactor, count)) {
    return false;
  }
  for (std::size_t index = 0u; index < count; ++index) {
    const ReactorWait &wait = ReactorRegistryWaitAt(reactor, index);
    PushWaitReady(reactor, wait, ReactorEventsForInterest(wait.interest),
                  ReactorReadyDisposition::PollFailed);
  }
  return true;
}

bool ReactorExpandInvalidAll(ReactorRuntime &reactor) noexcept {
  const std::size_t count = ReactorRegistrySize(reactor);
  if (!ReserveReady(reactor, count)) {
    return false;
  }
  for (std::size_t index = 0u; index < count; ++index) {
    const ReactorWait &wait = ReactorRegistryWaitAt(reactor, index);
    PushWaitReady(reactor, wait, ReactorEventsForInterest(wait.interest),
                  ReactorReadyDisposition::Invalid);
  }
  return true;
}

} // namespace rund::node
