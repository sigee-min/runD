#pragma once

#include "arithmetic.hpp"
#include "reservation.hpp"

namespace rund::node::accel::detail {

struct KernelExecutionStep;

} // namespace rund::node::accel::detail

namespace rund::node::accel::detail::backend_template_plan {

[[nodiscard]] std::uint64_t
primitive_pass_count(const KernelExecutionStep &step) noexcept;

} // namespace rund::node::accel::detail::backend_template_plan
