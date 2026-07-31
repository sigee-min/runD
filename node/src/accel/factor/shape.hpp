#pragma once

#include <kernel/program/compute/factor/plan.hpp>

#include "../kernel/bindings/factor.hpp"
#include "../primitive/shape.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] bool FactorShapeOk(const rund::kernel::FactorDesc &desc,
                                 const rund::kernel::FactorPlan &plan,
                                 const FactorBinds &bindings) noexcept;

} // namespace rund::node::accel::detail
