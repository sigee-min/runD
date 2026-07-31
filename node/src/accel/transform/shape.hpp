#pragma once

#include <kernel/program/compute/transform/plan.hpp>

#include "../primitive/shape.hpp"
#include "../kernel/bindings/transform.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] inline bool
TransformDescMatchesPlan(const rund::kernel::TransformDesc &desc,
                         const rund::kernel::TransformPlan &plan) noexcept {
  return rund::kernel::TransformPlanMatchesDesc(desc, plan) &&
         plan.layout == rund::kernel::TransformLayout::Split &&
         plan.element_count != 0u;
}

[[nodiscard]] bool
TransformShapeOk(const rund::kernel::TransformDesc &desc,
                 const rund::kernel::TransformPlan &plan,
                 const TransformBinds &bindings) noexcept;

} // namespace rund::node::accel::detail
