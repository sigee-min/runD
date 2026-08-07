#pragma once

#include <rund/task/stats/storage.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "model.hpp"

namespace rund::node {

enum class ReactorApplyDisposition : std::uint8_t {
  Success,
  Invalid,
  Failed,
  BackendUnavailable,
};

class ReactorApplyResult final {
public:
  [[nodiscard]] static constexpr ReactorApplyResult success() noexcept {
    return ReactorApplyResult{ReactorApplyDisposition::Success};
  }

  [[nodiscard]] static constexpr ReactorApplyResult
  invalid(const ReactorHandle handle) noexcept {
    return handle == kInvalidReactorHandle
               ? ReactorApplyResult{ReactorApplyDisposition::Failed}
               : ReactorApplyResult{ReactorApplyDisposition::Invalid, handle};
  }

  [[nodiscard]] static constexpr ReactorApplyResult failed() noexcept {
    return ReactorApplyResult{ReactorApplyDisposition::Failed};
  }

  [[nodiscard]] static constexpr ReactorApplyResult
  backend_unavailable() noexcept {
    return ReactorApplyResult{ReactorApplyDisposition::BackendUnavailable};
  }

  [[nodiscard]] constexpr ReactorApplyDisposition disposition() const noexcept {
    return disposition_;
  }

  [[nodiscard]] constexpr ReactorHandle invalid_handle() const noexcept {
    return invalid_handle_;
  }

private:
  constexpr explicit ReactorApplyResult(
      const ReactorApplyDisposition disposition) noexcept
      : disposition_(disposition) {}

  constexpr ReactorApplyResult(const ReactorApplyDisposition disposition,
                               const ReactorHandle invalid_handle) noexcept
      : disposition_(disposition), invalid_handle_(invalid_handle) {}

  ReactorApplyDisposition disposition_;
  ReactorHandle invalid_handle_ = kInvalidReactorHandle;
};

[[nodiscard]] ReactorApplyResult
ReactorBackendApplyChanges(ReactorRuntime &reactor,
                           ::rund::detail::task::StatStorage &stats) noexcept;

[[nodiscard]] ReactorPlatformPollResult
ReactorBackendPoll(ReactorRuntime &reactor,
                   ::rund::detail::task::StatStorage &stats, int timeout_ms,
                   std::size_t max_events) noexcept;

void ReactorCloseRuntime(ReactorRuntime &reactor) noexcept;

} // namespace rund::node
