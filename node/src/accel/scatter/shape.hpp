#pragma once

#include <kernel/program/compute/scatter/plan.hpp>

#include "../kernel/bindings/scatter.hpp"
#include "../primitive/shape.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] bool ScatterShapeOk(const rund::kernel::ScatterDesc &desc,
                                  const rund::kernel::ScatterPlan &plan,
                                  const ScatterBinds &bindings) noexcept;

} // namespace rund::node::accel::detail
