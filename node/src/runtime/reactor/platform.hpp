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

enum class ReactorPlatformOpDisposition : std::uint8_t {
  Success,
  Invalid,
  Failed,
  BackendUnavailable,
};

class ReactorPlatformOpResult final {
public:
  [[nodiscard]] static constexpr ReactorPlatformOpResult success() noexcept {
    return ReactorPlatformOpResult{ReactorPlatformOpDisposition::Success, 0};
  }

  [[nodiscard]] static constexpr ReactorPlatformOpResult
  invalid(const std::int64_t platform_error) noexcept {
    return ReactorPlatformOpResult{ReactorPlatformOpDisposition::Invalid,
                                   platform_error};
  }

  [[nodiscard]] static constexpr ReactorPlatformOpResult
  failed(const std::int64_t platform_error) noexcept {
    return ReactorPlatformOpResult{ReactorPlatformOpDisposition::Failed,
                                   platform_error};
  }

  [[nodiscard]] static constexpr ReactorPlatformOpResult
  backend_unavailable() noexcept {
    return ReactorPlatformOpResult{
        ReactorPlatformOpDisposition::BackendUnavailable, 0};
  }

  [[nodiscard]] constexpr ReactorPlatformOpDisposition
  disposition() const noexcept {
    return disposition_;
  }

  [[nodiscard]] constexpr std::int64_t platform_error() const noexcept {
    return platform_error_;
  }

private:
  constexpr ReactorPlatformOpResult(
      const ReactorPlatformOpDisposition disposition,
      const std::int64_t platform_error) noexcept
      : disposition_(disposition), platform_error_(platform_error) {}

  ReactorPlatformOpDisposition disposition_;
  std::int64_t platform_error_;
};

enum class ReactorPlatformBatchDisposition : std::uint8_t {
  Success,
  Invalid,
  Failed,
  BackendUnavailable,
};

class ReactorPlatformBatchResult final {
public:
  [[nodiscard]] static constexpr ReactorPlatformBatchResult success() noexcept {
    return ReactorPlatformBatchResult{ReactorPlatformBatchDisposition::Success,
                                      0, 0u};
  }

  [[nodiscard]] static constexpr ReactorPlatformBatchResult
  invalid(const std::int64_t platform_error,
          const std::size_t failed_index) noexcept {
    return ReactorPlatformBatchResult{ReactorPlatformBatchDisposition::Invalid,
                                      platform_error, failed_index};
  }

  [[nodiscard]] static constexpr ReactorPlatformBatchResult
  failed(const std::int64_t platform_error,
         const std::size_t failed_index) noexcept {
    return ReactorPlatformBatchResult{ReactorPlatformBatchDisposition::Failed,
                                      platform_error, failed_index};
  }

  [[nodiscard]] static constexpr ReactorPlatformBatchResult
  backend_unavailable() noexcept {
    return ReactorPlatformBatchResult{
        ReactorPlatformBatchDisposition::BackendUnavailable, 0, 0u};
  }

  [[nodiscard]] constexpr ReactorPlatformBatchDisposition
  disposition() const noexcept {
    return disposition_;
  }

  [[nodiscard]] constexpr std::int64_t platform_error() const noexcept {
    return platform_error_;
  }

  [[nodiscard]] constexpr std::size_t failed_index() const noexcept {
    return failed_index_;
  }

private:
  constexpr ReactorPlatformBatchResult(
      const ReactorPlatformBatchDisposition disposition,
      const std::int64_t platform_error,
      const std::size_t failed_index) noexcept
      : disposition_(disposition), platform_error_(platform_error),
        failed_index_(failed_index) {}

  ReactorPlatformBatchDisposition disposition_;
  std::int64_t platform_error_;
  std::size_t failed_index_;
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

enum class ReactorPlatformHandleIdentityDisposition : std::uint8_t {
  Invalid,
  Described,
};

class ReactorPlatformHandleIdentity final {
public:
  [[nodiscard]] static constexpr ReactorPlatformHandleIdentity
  invalid() noexcept {
    return ReactorPlatformHandleIdentity{
        ReactorPlatformHandleIdentityDisposition::Invalid, 0u, 0u, 0u};
  }

  [[nodiscard]] static constexpr ReactorPlatformHandleIdentity
  described(const std::uint64_t device, const std::uint64_t inode,
            const std::uint32_t mode) noexcept {
    return ReactorPlatformHandleIdentity{
        ReactorPlatformHandleIdentityDisposition::Described, device, inode,
        mode};
  }

  [[nodiscard]] constexpr ReactorPlatformHandleIdentityDisposition
  disposition() const noexcept {
    return disposition_;
  }

  [[nodiscard]] constexpr std::uint64_t device() const noexcept {
    return device_;
  }

  [[nodiscard]] constexpr std::uint64_t inode() const noexcept {
    return inode_;
  }

  [[nodiscard]] constexpr std::uint32_t mode() const noexcept { return mode_; }

  [[nodiscard]] constexpr bool same_object(
      const ReactorPlatformHandleIdentity &other) const noexcept {
    return disposition_ == ReactorPlatformHandleIdentityDisposition::Described &&
           other.disposition_ ==
               ReactorPlatformHandleIdentityDisposition::Described &&
           device_ == other.device_ && inode_ == other.inode_ &&
           mode_ == other.mode_;
  }

private:
  constexpr ReactorPlatformHandleIdentity(
      const ReactorPlatformHandleIdentityDisposition disposition,
      const std::uint64_t device, const std::uint64_t inode,
      const std::uint32_t mode) noexcept
      : device_(device), inode_(inode), mode_(mode),
        disposition_(disposition) {}

  std::uint64_t device_;
  std::uint64_t inode_;
  std::uint32_t mode_;
  ReactorPlatformHandleIdentityDisposition disposition_;
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
                    std::size_t max_events,
                    std::vector<ReactorPlatformReady> &out) noexcept;
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
