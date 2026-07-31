#include <cluster/retry/identity.hpp>

namespace rund::cluster {

RetryDecision evaluate_retry(const RetryRequest &request) {
  const RunAttempt attempt{request.next, request.retry};
  if (!request.previous.complete() || !request.next.complete()) {
    return RetryDecision{.code = RetryCode::KeyIncomplete, .attempt = attempt};
  }
  if (request.previous != request.next) {
    return RetryDecision{.code = RetryCode::IdentityChanged,
                         .attempt = attempt};
  }
  return RetryDecision{.code = RetryCode::IdentityPreserved,
                       .attempt = attempt};
}

} // namespace rund::cluster
