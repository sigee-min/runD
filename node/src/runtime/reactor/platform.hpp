#pragma once

// The complete platform-neutral scheduler-to-reactor backend contract.

#include "readiness/state.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace rund::node {

struct ReactorPlatformReady {
  ReactorHandle handle = kInvalidReactorHandle;
  ReactorEvent events = ReactorEvent::None;
  bool invalid = false;
};

struct ReactorPlatformRegistration {
  ReactorHandle handle = kInvalidReactorHandle;
  ReactorInterest interest = ReactorInterest::None;
};

struct ReactorPlatformState;

struct ReactorPlatformStateDelete {
  void operator()(ReactorPlatformState *state) const noexcept;
};

struct ReactorPlatform {
  std::unique_ptr<ReactorPlatformState, ReactorPlatformStateDelete> state{};
};

struct ReactorPlatformOpResult {
  bool ok = true;
  bool invalid = false;
  bool unavailable = false;
  std::int64_t platform_error = 0u;
};

struct ReactorPlatformBatchResult {
  bool ok = true;
  bool invalid = false;
  bool unavailable = false;
  std::int64_t platform_error = 0u;
  std::size_t failed_index = 0u;
};

struct ReactorPlatformPollResult {
  bool ok = true;
  bool invalid = false;
  bool unavailable = false;
  std::int64_t platform_error = 0u;
  const std::vector<ReactorPlatformReady> *ready = nullptr;
};

struct ReactorPlatformHandleIdentity {
  std::uint64_t device = 0u;
  std::uint64_t inode = 0u;
  std::uint32_t mode = 0u;
  bool valid = false;
};

struct ReactorRegistrationChange {
  enum class Kind : std::uint8_t {
    Add,
    Modify,
    Remove,
  };

  Kind kind = Kind::Add;
  ReactorHandle handle = kInvalidReactorHandle;
  ReactorInterest interest = ReactorInterest::None;
  std::uint64_t fd_generation = 0u;
  bool best_effort = false;
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

[[nodiscard]] ReactorPlatformOpResult
OpenReactorPlatform(ReactorPlatform &platform) noexcept;
[[nodiscard]] ReactorPlatformOpResult
PrepareReactorPlatform(ReactorPlatform &platform,
                       std::size_t capacity) noexcept;
void CloseReactorPlatform(ReactorPlatform &platform) noexcept;

[[nodiscard]] ReactorPlatformOpResult
AddReactorPlatformInterest(ReactorPlatform &platform, ReactorHandle handle,
                           ReactorInterest interest) noexcept;
[[nodiscard]] ReactorPlatformOpResult
ModifyReactorPlatformInterest(ReactorPlatform &platform, ReactorHandle handle,
                              ReactorInterest interest) noexcept;
[[nodiscard]] ReactorPlatformOpResult
RemoveReactorPlatformInterest(ReactorPlatform &platform,
                              ReactorHandle handle) noexcept;

[[nodiscard]] ReactorPlatformBatchResult
ApplyReactorPlatformChanges(ReactorPlatform &platform,
                            const ReactorRegistrationChange *changes,
                            std::size_t count) noexcept;
[[nodiscard]] ReactorPlatformPollResult
PollReactorPlatform(ReactorPlatform &platform, int timeout_ms,
                    std::size_t max_events) noexcept;
[[nodiscard]] BatchIoProbeResult
ProbeReactorPlatformNow(ReactorPlatform &platform,
                        const BatchIoPollRequest *requests, std::size_t count,
                        std::vector<BatchIoReady> &out) noexcept;
[[nodiscard]] ReactorPlatformHandleIdentity
DescribeReactorPlatformHandle(ReactorHandle handle) noexcept;
[[nodiscard]] ReactorHandle
RetainReactorPlatformHandle(ReactorHandle handle) noexcept;
void ReleaseReactorPlatformHandle(ReactorHandle handle) noexcept;

} // namespace rund::node
