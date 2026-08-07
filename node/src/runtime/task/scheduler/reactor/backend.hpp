#pragma once

#include <rund/task/stats/storage.hpp>

#include <cstdint>

#include "change/token.hpp"

namespace rund::node {

struct ReactorRuntime;

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
  invalid(const ReactorHandle handle,
          const std::uint64_t fd_generation) noexcept {
    const ReactorInvalidChangeToken token =
        ReactorInvalidChangeToken::observed(handle, fd_generation);
    return token.valid()
               ? ReactorApplyResult{ReactorApplyDisposition::Invalid, token}
               : ReactorApplyResult{ReactorApplyDisposition::Failed};
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

  [[nodiscard]] constexpr ReactorInvalidChangeToken
  invalid_change() const noexcept {
    return invalid_change_;
  }

private:
  constexpr explicit ReactorApplyResult(
      const ReactorApplyDisposition disposition) noexcept
      : disposition_(disposition) {}

  constexpr ReactorApplyResult(
      const ReactorApplyDisposition disposition,
      const ReactorInvalidChangeToken invalid_change) noexcept
      : disposition_(disposition), invalid_change_(invalid_change) {}

  ReactorApplyDisposition disposition_;
  ReactorInvalidChangeToken invalid_change_ = ReactorInvalidChangeToken::none();
};

[[nodiscard]] constexpr bool
ReactorApplyAllowsLogicalProgress(const ReactorApplyResult result) noexcept {
  return result.disposition() == ReactorApplyDisposition::Success ||
         result.disposition() == ReactorApplyDisposition::Invalid;
}

[[nodiscard]] ReactorApplyResult
ReactorBackendApplyChanges(ReactorRuntime &reactor,
                           ::rund::detail::task::StatStorage &stats) noexcept;

void ReactorCloseRuntime(ReactorRuntime &reactor) noexcept;

} // namespace rund::node
