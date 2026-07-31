#include <rund/session/trace.hpp>

namespace rund {
namespace {

[[nodiscard]] std::string_view
ComputeError(const ::rund::compute::Reason reason) noexcept {
  switch (reason) {
#define RUND_COMPUTE_REASON(name, value, text)                                 \
  case ::rund::compute::Reason::name:                                          \
    return text;
#include <rund/compute/reason.def>
#undef RUND_COMPUTE_REASON
  }
  return "compute_reason_invalid";
}

} // namespace

bool TraceCode::valid() const noexcept {
  switch (domain_) {
  case TraceDomain::Runtime:
    return ValidReasonCode(static_cast<ReasonCode>(value_));
  case TraceDomain::Compute: {
    const auto reason = static_cast<::rund::compute::Reason>(value_);
    return reason == ::rund::compute::Reason::Ok ||
           ::rund::compute::detail::valid(reason);
  }
  }
  return false;
}

std::string_view TraceCode::error() const noexcept {
  if (!valid()) {
    return "trace_code_invalid";
  }
  if (ok()) {
    return {};
  }
  if (domain_ == TraceDomain::Runtime) {
    return ReasonString(static_cast<ReasonCode>(value_));
  }
  return ComputeError(static_cast<::rund::compute::Reason>(value_));
}

} // namespace rund
