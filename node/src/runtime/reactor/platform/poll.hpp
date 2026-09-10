#pragma once

#include "../readiness/state.hpp"
#include "result.hpp"
#include "state.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rund::node {

struct ReactorPlatformReady {
  ReactorHandle handle = kInvalidReactorHandle;
  ReactorEvent events = ReactorEvent::None;
  bool invalid = false;
};

enum class ReactorPlatformPollDisposition : std::uint8_t {
  Success,
  Invalid,
  Failed,
  BackendUnavailable,
};

class ReactorPlatformPollResult final {
public:
  [[nodiscard]] static constexpr ReactorPlatformPollResult success() noexcept {
    return ReactorPlatformPollResult{ReactorPlatformPollDisposition::Success,
                                     0};
  }

  [[nodiscard]] static constexpr ReactorPlatformPollResult
  invalid(const std::int64_t platform_error) noexcept {
    return ReactorPlatformPollResult{ReactorPlatformPollDisposition::Invalid,
                                     platform_error};
  }

  [[nodiscard]] static constexpr ReactorPlatformPollResult
  failed(const std::int64_t platform_error) noexcept {
    return ReactorPlatformPollResult{ReactorPlatformPollDisposition::Failed,
                                     platform_error};
  }

  [[nodiscard]] static constexpr ReactorPlatformPollResult
  backend_unavailable() noexcept {
    return ReactorPlatformPollResult{
        ReactorPlatformPollDisposition::BackendUnavailable, 0};
  }

  [[nodiscard]] constexpr ReactorPlatformPollDisposition
  disposition() const noexcept {
    return disposition_;
  }

  [[nodiscard]] constexpr std::int64_t platform_error() const noexcept {
    return platform_error_;
  }

private:
  constexpr ReactorPlatformPollResult(
      const ReactorPlatformPollDisposition disposition,
      const std::int64_t platform_error) noexcept
      : disposition_(disposition), platform_error_(platform_error) {}

  ReactorPlatformPollDisposition disposition_;
  std::int64_t platform_error_;
};

struct BatchIoPollRequest {
  std::uint32_t index = 0u;
  ReactorHandle handle = kInvalidReactorHandle;
  ReactorInterest interest = ReactorInterest::None;
};

struct BatchIoReady {
  std::uint32_t index = 0u;
  ReactorEvent events = ReactorEvent::None;
  bool invalid = false;
};

enum class BatchIoProbeDisposition : std::uint8_t {
  Success,
  Failed,
  BackendUnavailable,
};

class BatchIoProbeResult final {
public:
  [[nodiscard]] static constexpr BatchIoProbeResult success() noexcept {
    return BatchIoProbeResult{BatchIoProbeDisposition::Success, 0};
  }

  [[nodiscard]] static constexpr BatchIoProbeResult
  failed(const std::int64_t platform_error) noexcept {
    return BatchIoProbeResult{BatchIoProbeDisposition::Failed, platform_error};
  }

  [[nodiscard]] static constexpr BatchIoProbeResult
  backend_unavailable() noexcept {
    return BatchIoProbeResult{BatchIoProbeDisposition::BackendUnavailable, 0};
  }

  [[nodiscard]] constexpr BatchIoProbeDisposition disposition() const noexcept {
    return disposition_;
  }

  [[nodiscard]] constexpr std::int64_t platform_error() const noexcept {
    return platform_error_;
  }

private:
  constexpr BatchIoProbeResult(const BatchIoProbeDisposition disposition,
                               const std::int64_t platform_error) noexcept
      : disposition_(disposition), platform_error_(platform_error) {}

  BatchIoProbeDisposition disposition_;
  std::int64_t platform_error_;
};

[[nodiscard]] ReactorPlatformPollResult
PollReactorPlatform(ReactorPlatform &, int, std::size_t,
                    std::vector<ReactorPlatformReady> &) noexcept;
[[nodiscard]] BatchIoProbeResult
ProbeReactorPlatformNow(ReactorPlatform &, const BatchIoPollRequest *,
                        std::size_t, std::vector<BatchIoReady> &) noexcept;

} // namespace rund::node
