#pragma once

#include <cluster/run/identity.hpp>

#include <cstdint>
#include <string_view>

namespace rund::cluster {

struct RetryRequest {
  RunKey previous{};
  RunKey next{};
  RetryEpoch retry{};
};

enum class RetryCode : std::uint8_t {
  NotEvaluated = 0u,
  KeyIncomplete = 1u,
  IdentityChanged = 2u,
  IdentityPreserved = 3u,
};

struct RetryDecision {
  RetryCode code = RetryCode::NotEvaluated;
  RunAttempt attempt{};

  [[nodiscard]] constexpr bool preserves_identity() const noexcept {
    return code == RetryCode::IdentityPreserved;
  }
  [[nodiscard]] constexpr std::string_view reason() const noexcept {
    switch (code) {
    case RetryCode::NotEvaluated:
      return "not_evaluated";
    case RetryCode::KeyIncomplete:
      return "run_key_incomplete";
    case RetryCode::IdentityChanged:
      return "run_identity_changed";
    case RetryCode::IdentityPreserved:
      return "run_identity_preserved";
    }
    return "retry_code_invalid";
  }
};

[[nodiscard]] RetryDecision evaluate_retry(const RetryRequest &request);

} // namespace rund::cluster
