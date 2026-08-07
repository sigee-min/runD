#include "local.hpp"
#include "../../../reactor/diagnostics.hpp"

#if defined(__APPLE__) || defined(__FreeBSD__)

namespace rund::node {

ReactorPlatformOpResult AddReactorPlatformInterest(
    ReactorPlatform& platform, const ReactorHandle handle,
    const ReactorInterest interest) noexcept {
  if (!KqueueReserveInterestStorage(platform, handle)) {
    return ReactorPlatformOpResult::failed(ENOMEM);
  }
  const ReactorPlatformOpResult added =
      KqueueAddInterest(platform, handle, interest);
  if (added.disposition() == ReactorPlatformOpDisposition::Success) {
    if (!KqueueRememberInterest(platform, handle, interest)) {
      return ReactorPlatformOpResult::failed(ENOMEM);
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
    return ReactorPlatformOpResult::failed(ENOMEM);
  }
  const ReactorPlatformOpResult removed =
      KqueueRemoveInterest(platform, handle, true);
  if (removed.disposition() != ReactorPlatformOpDisposition::Success &&
      removed.disposition() != ReactorPlatformOpDisposition::Invalid) {
    return removed;
  }
  const ReactorPlatformOpResult added =
      KqueueAddInterest(platform, handle, interest);
  if (added.disposition() == ReactorPlatformOpDisposition::Success &&
      !KqueueRememberInterest(platform, handle, interest)) {
    return ReactorPlatformOpResult::failed(ENOMEM);
  }
  return added;
}

ReactorPlatformOpResult RemoveReactorPlatformInterest(
    ReactorPlatform& platform, const ReactorHandle handle) noexcept {
  const ReactorPlatformOpResult removed =
      KqueueRemoveInterest(platform, handle, true);
  if (removed.disposition() == ReactorPlatformOpDisposition::Success ||
      removed.disposition() == ReactorPlatformOpDisposition::Invalid) {
    KqueueForgetInterest(platform, handle);
    RecordReactorPlatformRemove();
  }
  return removed;
}

}  // namespace rund::node

#endif
