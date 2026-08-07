#pragma once

#include "../../../reactor/platform.hpp"

#include <cstdint>
#include <vector>

namespace rund::node {

enum class ReactorProbeDisposition : std::uint8_t {
  NotReady,
  Ready,
  Invalid,
  PollFailed,
  BackendUnavailable,
};

class ReactorProbeResult final {
public:
  [[nodiscard]] static constexpr ReactorProbeResult not_ready() noexcept {
    return ReactorProbeResult{ReactorProbeDisposition::NotReady};
  }

  [[nodiscard]] static constexpr ReactorProbeResult
  ready(const ReactorEvent events) noexcept {
    return ReactorProbeResult{ReactorProbeDisposition::Ready, events};
  }

  [[nodiscard]] static constexpr ReactorProbeResult
  invalid(const ReactorEvent events) noexcept {
    return ReactorProbeResult{ReactorProbeDisposition::Invalid, events};
  }

  [[nodiscard]] static constexpr ReactorProbeResult poll_failed() noexcept {
    return ReactorProbeResult{ReactorProbeDisposition::PollFailed};
  }

  [[nodiscard]] static constexpr ReactorProbeResult
  backend_unavailable() noexcept {
    return ReactorProbeResult{ReactorProbeDisposition::BackendUnavailable};
  }

  [[nodiscard]] constexpr ReactorProbeDisposition disposition() const noexcept {
    return disposition_;
  }

  [[nodiscard]] constexpr ReactorEvent events() const noexcept {
    return events_;
  }

private:
  constexpr explicit ReactorProbeResult(
      const ReactorProbeDisposition disposition) noexcept
      : disposition_(disposition) {}

  constexpr ReactorProbeResult(const ReactorProbeDisposition disposition,
                               const ReactorEvent events) noexcept
      : disposition_(disposition), events_(events) {}

  ReactorProbeDisposition disposition_;
  ReactorEvent events_ = ReactorEvent::None;
};

[[nodiscard]] ReactorProbeResult
ReactorProbeNow(ReactorPlatform &platform, std::vector<BatchIoReady> &scratch,
                ReactorHandle handle, ReactorInterest interest) noexcept;

} // namespace rund::node
