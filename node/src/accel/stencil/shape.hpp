#pragma once

#include "../kernel/bindings/stencil.hpp"
#include "../primitive/shape.hpp"

#include <kernel/program/compute/stencil/plan.hpp>

namespace rund::node::accel::detail {

[[nodiscard]] bool StencilShapeOk(const rund::kernel::StencilDesc &desc,
                                  const rund::kernel::StencilPlan &plan,
                                  const StencilBinds &bindings) noexcept;

} // namespace rund::node::accel::detail
