#pragma once

#include <cstddef>
#include <cstdint>

namespace rund::node {

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

} // namespace rund::node
