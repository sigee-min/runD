#include "local.hpp"
#include "../../../reactor/diagnostics.hpp"

#if defined(__APPLE__) || defined(__FreeBSD__)

namespace rund::node {

ReactorPlatformOpResult AddReactorPlatformInterest(
    ReactorPlatform& platform, const ReactorHandle handle,
    const ReactorInterest interest) noexcept {
  if (!KqueueReserveInterestStorage(platform, handle)) {
    return ReactorPlatformOpResult{.ok = false, .platform_error = ENOMEM};
  }
  const ReactorPlatformOpResult added =
      KqueueAddInterest(platform, handle, interest);
  if (added.ok) {
    if (!KqueueRememberInterest(platform, handle, interest)) {
      return ReactorPlatformOpResult{.ok = false, .platform_error = ENOMEM};
    }
    RecordReactorPlatformAdd();
  }
  return added;
}

ReactorPlatformOpResult ModifyReactorPlatformInterest(
    ReactorPlatform& platform, const ReactorHandle handle,
    const ReactorInterest interest) noexcept {
  RecordReactorPlatformModify();
  if (!KqueueReserveInterestStorage(platform, handle)) {
    return ReactorPlatformOpResult{.ok = false, .platform_error = ENOMEM};
  }
  const ReactorPlatformOpResult removed =
      KqueueRemoveInterest(platform, handle, true);
  if (!removed.ok && !removed.invalid) {
    return removed;
  }
  const ReactorPlatformOpResult added =
      KqueueAddInterest(platform, handle, interest);
  if (added.ok && !KqueueRememberInterest(platform, handle, interest)) {
    return ReactorPlatformOpResult{.ok = false, .platform_error = ENOMEM};
  }
  return added;
}

ReactorPlatformOpResult RemoveReactorPlatformInterest(
    ReactorPlatform& platform, const ReactorHandle handle) noexcept {
  const ReactorPlatformOpResult removed =
      KqueueRemoveInterest(platform, handle, true);
  if (removed.ok || removed.invalid) {
    KqueueForgetInterest(platform, handle);
    RecordReactorPlatformRemove();
  }
  return removed;
}

}  // namespace rund::node

#endif
