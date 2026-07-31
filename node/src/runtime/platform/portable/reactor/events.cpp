#include "local.hpp"

#include "../../../reactor/readiness/handle.hpp"

#if !defined(__APPLE__) && !defined(__FreeBSD__) && !defined(__linux__)

namespace rund::node {

bool PollReadyEvent(const pollfd& descriptor,
                    ReactorPlatformReady* const out) noexcept {
  if (descriptor.revents == 0) return false;
  const bool invalid = IoPollInvalid(descriptor.revents);
  const ReactorEvent events = PosixEvents(descriptor.revents);
  if (!invalid && events == ReactorEvent::None) return false;
  if (out != nullptr) {
    *out = ReactorPlatformReady{
        .handle = ReactorHandleFromPublic(descriptor.fd),
        .events = events,
        .invalid = invalid,
    };
  }
  return true;
}

}  // namespace rund::node

#endif
