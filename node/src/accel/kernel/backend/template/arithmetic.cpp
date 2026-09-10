#include "arithmetic.hpp"

#include <kernel/core/checked.hpp>

namespace rund::node::accel::detail::backend_template_plan {

bool add(std::uint64_t &target, const std::uint64_t value) noexcept {
  return rund::kernel::checked::add(target, value, target);
}

bool product(const std::uint64_t left, const std::uint64_t right,
             std::uint64_t &out) noexcept {
  return rund::kernel::checked::mul(left, right, out);
}

} // namespace rund::node::accel::detail::backend_template_plan
