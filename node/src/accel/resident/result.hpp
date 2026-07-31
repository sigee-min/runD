#pragma once

#include <accel/check.hpp>

namespace rund::node::accel::detail {

template <typename Result>
[[nodiscard]] inline Result RejectResident(const char *const reason) {
  return Result{.check = rund::AccelCheck{false, reason}};
}

} // namespace rund::node::accel::detail
