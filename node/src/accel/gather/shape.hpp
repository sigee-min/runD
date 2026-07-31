#pragma once

#include "../kernel/bindings/gather.hpp"
#include "../primitive/shape.hpp"

#include <kernel/program/compute/gather/plan.hpp>

namespace rund::node::accel::detail {

[[nodiscard]] bool GatherShapeOk(const rund::kernel::GatherDesc &desc,
                                 const rund::kernel::GatherPlan &plan,
                                 const GatherBinds &bindings) noexcept;

} // namespace rund::node::accel::detail
