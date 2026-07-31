#pragma once

#include <kernel/program/compute/solve/plan.hpp>

#include "../kernel/bindings/solve.hpp"
#include "../primitive/shape.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] bool SolveShapeOk(const rund::kernel::SolveDesc &desc,
                                const rund::kernel::SolvePlan &plan,
                                const SolveBinds &bindings) noexcept;

} // namespace rund::node::accel::detail
