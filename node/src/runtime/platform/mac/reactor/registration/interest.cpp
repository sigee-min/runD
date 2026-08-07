#include "../local.hpp"
#include "../../../../reactor/diagnostics.hpp"

#if defined(__APPLE__) || defined(__FreeBSD__)

#include <algorithm>

namespace rund::node {

bool KqueueReserveInterestStorage(ReactorPlatform& platform,
                                  const ReactorHandle handle) noexcept {
  for (const ReactorPlatformRegistration& registration :
       MacReactorState(platform).registrations) {
    if (registration.handle == handle) return true;
  }
  try {
    MacReactorState(platform).registrations.reserve(
        MacReactorState(platform).registrations.size() + 1u);
  } catch (...) {
    return false;
  }
  return true;
}

bool KqueueRememberInterest(ReactorPlatform& platform,
                            const ReactorHandle handle,
                            const ReactorInterest interest) noexcept {
  for (ReactorPlatformRegistration& registration :
       MacReactorState(platform).registrations) {
    if (registration.handle == handle) {
      registration.interest = interest;
      return true;
    }
  }
  try {
    MacReactorState(platform).registrations.push_back(
        ReactorPlatformRegistration{.handle = handle, .interest = interest});
  } catch (...) {
    return false;
  }
  return true;
}

void KqueueForgetInterest(ReactorPlatform& platform,
                          const ReactorHandle handle) noexcept {
  const auto found =
      std::find_if(MacReactorState(platform).registrations.begin(),
                   MacReactorState(platform).registrations.end(),
                   [handle](const ReactorPlatformRegistration& registration) {
                     return registration.handle == handle;
                   });
  if (found != MacReactorState(platform).registrations.end()) {
    MacReactorState(platform).registrations.erase(found);
  }
}

bool KqueueReserveBatchInterestStorage(
    ReactorPlatform& platform, const ReactorRegistrationChange* const changes,
    const std::size_t count) noexcept {
  for (std::size_t index = 0u; index < count; ++index) {
    const ReactorRegistrationChange& change = changes[index];
    if ((change.kind() == ReactorRegistrationChange::Kind::Add ||
         change.kind() == ReactorRegistrationChange::Kind::Modify) &&
        !KqueueReserveInterestStorage(platform, change.handle())) {
      return false;
    }
  }
  return true;
}

void KqueueRememberBatchInterest(ReactorPlatform& platform,
                                 const ReactorRegistrationChange* const changes,
                                 const std::size_t count) noexcept {
  for (std::size_t index = 0u; index < count; ++index) {
    const ReactorRegistrationChange& change = changes[index];
    switch (change.kind()) {
      case ReactorRegistrationChange::Kind::Add:
        (void)KqueueRememberInterest(platform, change.handle(),
                                     change.interest());
        RecordReactorPlatformAdd();
        break;
      case ReactorRegistrationChange::Kind::Modify:
        RecordReactorPlatformModify();
        (void)KqueueRememberInterest(platform, change.handle(),
                                     change.interest());
        break;
      case ReactorRegistrationChange::Kind::CleanupRemove:
        KqueueForgetInterest(platform, change.handle());
        RecordReactorPlatformRemove();
        break;
    }
  }
}

}  // namespace rund::node

#endif
