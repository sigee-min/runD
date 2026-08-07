#include "probe.hpp"

#include <cerrno>

#include "../io.hpp"

namespace rund::node {

BatchIoProbeResult PollPosixReadyNow(const BatchIoPollRequest* const requests,
                                     const std::size_t count,
                                     std::vector<BatchIoReady>& out,
                                     PosixProbeBuffer& buffer) noexcept {
  out.clear();
  buffer.events.clear();
  if (requests == nullptr || count == 0u) {
    return BatchIoProbeResult::success();
  }

  try {
    buffer.events.reserve(count);
    out.reserve(count);
    for (std::size_t index = 0u; index < count; ++index) {
      buffer.events.push_back(pollfd{
          .fd = PosixFd(requests[index].handle),
          .events = PosixInterest(requests[index].interest),
          .revents = 0,
      });
    }
  } catch (...) {
    out.clear();
    buffer.events.clear();
    return BatchIoProbeResult::failed(ENOMEM);
  }

  int native = 0;
  do {
    errno = 0;
    native = ::poll(buffer.events.data(), buffer.events.size(), 0);
  } while (native < 0 && errno == EINTR);
  if (native < 0) {
    out.clear();
    return BatchIoProbeResult::failed(errno);
  }

  try {
    for (std::size_t index = 0u; index < count; ++index) {
      const short revents = buffer.events[index].revents;
      if (revents == 0) {
        continue;
      }
      const bool invalid = IoPollInvalid(revents);
      const ReactorEvent ready = PosixEvents(revents);
      if (!invalid && ready == ReactorEvent::None) {
        continue;
      }
      out.push_back(BatchIoReady{
          .index = requests[index].index,
          .events = ready,
          .invalid = invalid,
      });
    }
  } catch (...) {
    out.clear();
    return BatchIoProbeResult::failed(ENOMEM);
  }

  return BatchIoProbeResult::success();
}

}  // namespace rund::node
