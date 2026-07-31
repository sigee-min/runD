#include <accel/buffer.hpp>
#include <accel/check.hpp>

#include "shared.hpp"

namespace rund::node::accel::detail {

rund::AccelCheck RejectAccelCheck(const char *const reason) noexcept {
  return rund::AccelCheck{false, reason};
}

rund::AccelCheck OkAccelCheck() noexcept {
  return rund::AccelCheck{true, "ok"};
}

} // namespace rund::node::accel::detail
