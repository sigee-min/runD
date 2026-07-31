#pragma once

#include "../kernel/bindings/sort.hpp"
#include "../primitive/shape.hpp"

#include <kernel/program/compute/sort/plan.hpp>

namespace rund::node::accel::detail {

[[nodiscard]] bool SortShapeOk(const rund::kernel::SortDesc &desc,
                               const rund::kernel::SortPlan &plan,
                               const SortBinds &bindings) noexcept;

} // namespace rund::node::accel::detail
