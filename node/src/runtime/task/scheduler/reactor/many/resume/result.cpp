#include "local.hpp"

namespace rund::node {

ReasonCode ReadyManyResumeResultCode(const ReasonCode ready_code,
                                     const bool cleanup_ok) noexcept {
  if (ready_code == ReasonCode::Ok && !cleanup_ok) {
    return ReasonCode::IoPollFailed;
  }
  return ready_code;
}

::rund::net::ready::many::Result
MakeReadyManyResumeResult(const ReasonCode result_code,
                          const std::uint32_t copied,
                          const bool budget_exhausted) noexcept {
  ::rund::net::ready::many::Result result{result_code};
  result.events = copied;
  result.budget_exhausted = budget_exhausted;
  return result;
}

} // namespace rund::node
