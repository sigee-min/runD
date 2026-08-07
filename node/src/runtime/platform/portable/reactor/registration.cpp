#include "local.hpp"
#include "../../../reactor/diagnostics.hpp"

#if !defined(__APPLE__) && !defined(__FreeBSD__) && !defined(__linux__)

#include <algorithm>
#include <cerrno>

namespace rund::node {

bool PollReserveInterestStorage(ReactorPlatform& platform,
                                const ReactorHandle handle) noexcept {
  for (const ReactorPlatformRegistration& registration :
       PortableReactorState(platform).registrations) {
    if (registration.handle == handle) return true;
  }
  try {
    PortableReactorState(platform).registrations.reserve(
        PortableReactorState(platform).registrations.size() + 1u);
  } catch (...) {
    return false;
  }
  return true;
}

bool PollRememberInterest(ReactorPlatform& platform, const ReactorHandle handle,
                          const ReactorInterest interest) noexcept {
  for (ReactorPlatformRegistration& registration :
       PortableReactorState(platform).registrations) {
    if (registration.handle == handle) {
      registration.interest = interest;
      return true;
    }
  }
  try {
    PortableReactorState(platform).registrations.push_back(
        ReactorPlatformRegistration{.handle = handle, .interest = interest});
  } catch (...) {
    return false;
  }
  return true;
}

void PollForgetInterest(ReactorPlatform& platform,
                        const ReactorHandle handle) noexcept {
  const auto found =
      std::find_if(PortableReactorState(platform).registrations.begin(),
                   PortableReactorState(platform).registrations.end(),
                   [handle](const ReactorPlatformRegistration& registration) {
                     return registration.handle == handle;
                   });
  if (found != PortableReactorState(platform).registrations.end()) {
    PortableReactorState(platform).registrations.erase(found);
  }
}

ReactorPlatformOpResult AddReactorPlatformInterest(
    ReactorPlatform& platform, const ReactorHandle handle,
    const ReactorInterest interest) noexcept {
  if (!PollReserveInterestStorage(platform, handle) ||
      !PollRememberInterest(platform, handle, interest)) {
    return ReactorPlatformOpResult{.ok = false, .platform_error = ENOMEM};
  }
  RecordReactorPlatformAdd();
  return {};
}

ReactorPlatformOpResult ModifyReactorPlatformInterest(
    ReactorPlatform& platform, const ReactorHandle handle,
    const ReactorInterest interest) noexcept {
  RecordReactorPlatformModify();
  if (!PollReserveInterestStorage(platform, handle) ||
      !PollRememberInterest(platform, handle, interest)) {
    return ReactorPlatformOpResult{.ok = false, .platform_error = ENOMEM};
  }
  return {};
}

ReactorPlatformOpResult RemoveReactorPlatformInterest(
    ReactorPlatform& platform, const ReactorHandle handle) noexcept {
  PollForgetInterest(platform, handle);
  RecordReactorPlatformRemove();
  return {};
}

ReactorPlatformBatchResult ApplyReactorPlatformChanges(
    ReactorPlatform& platform, const ReactorRegistrationChange* const changes,
    const std::size_t count) noexcept {
  if (changes == nullptr || count == 0u) {
    return ReactorPlatformBatchResult::success();
  }
  for (std::size_t index = 0u; index < count; ++index) {
    ReactorPlatformOpResult result{};
    const ReactorRegistrationChange& change = changes[index];
    switch (change.kind) {
      case ReactorRegistrationChange::Kind::Add:
        result = AddReactorPlatformInterest(platform, change.handle,
                                            change.interest);
        break;
      case ReactorRegistrationChange::Kind::Modify:
        result = ModifyReactorPlatformInterest(platform, change.handle,
                                               change.interest);
        break;
      case ReactorRegistrationChange::Kind::Remove:
        result = RemoveReactorPlatformInterest(platform, change.handle);
        break;
    }
    if (!result.ok) {
      return result.invalid
                 ? ReactorPlatformBatchResult::invalid(result.platform_error,
                                                       index)
                 : ReactorPlatformBatchResult::failed(result.platform_error,
                                                      index);
    }
  }
  return ReactorPlatformBatchResult::success();
}

}  // namespace rund::node

#endif
