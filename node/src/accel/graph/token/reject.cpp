#include <accel/kernel/check.hpp>

#include "local.hpp"

namespace rund::node::accel::detail {

KernelAdmission RejectAdmission(const char *const reason) noexcept {
  return KernelAdmission{
      .check = rund::AccelKernelCheck{false, reason},
  };
}

} // namespace rund::node::accel::detail
