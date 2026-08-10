#pragma once

#include "../kernel/bindings/range.hpp"

#include <kernel/program/compute/stencil/plan.hpp>

namespace rund::node::accel::detail {

[[nodiscard]] bool StencilBindingsMatch(const rund::kernel::StencilDesc &desc,
                                        const rund::kernel::StencilPlan &plan,
                                        const RangeBinds &bindings) noexcept;

} // namespace rund::node::accel::detail
