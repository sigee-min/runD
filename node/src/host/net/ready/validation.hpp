#pragma once

#include "../interest.hpp"
#include "../registry/socket.hpp"

#include <span>

namespace rund::net::ready::many::validation {

[[nodiscard]] inline ready::many::Result
Fail(const ::rund::ReasonCode code) noexcept {
  return ready::many::Result{code};
}

[[nodiscard]] inline ready::many::Result
Shape(const std::span<const ready::Request> requests,
      const std::span<ready::Event> out,
      const ready::many::Budget budget) noexcept {
  if (requests.empty()) {
    return ready::many::Result{::rund::ReasonCode::Ok};
  }
  if (out.empty()) {
    return Fail(::rund::ReasonCode::TaskInvalid);
  }
  if (budget.max_events == 0u) {
    ready::many::Result result{::rund::ReasonCode::Ok};
    result.budget_exhausted = true;
    return result;
  }
  for (const ready::Request &request : requests) {
    if (ReactorInterestFor(request.interest) == node::ReactorInterest::None) {
      return Fail(::rund::ReasonCode::TaskInvalid);
    }
  }
  return ready::many::Result{::rund::ReasonCode::Ok};
}

[[nodiscard]] inline bool
Current(const std::span<const ready::Request> requests) noexcept {
  for (const ready::Request &request : requests) {
    if (!IsCurrentSocket(request.socket)) {
      return false;
    }
  }
  return true;
}

} // namespace rund::net::ready::many::validation
