#pragma once

#include <kernel/program/compute/matrix/plan.hpp>

#include "../kernel/bindings/matrix.hpp"
#include "../primitive/shape.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] bool MatrixShapeOk(const rund::kernel::MatrixDesc &desc,
                                 const rund::kernel::MatrixPlan &plan,
                                 const MatrixBinds &bindings) noexcept;

} // namespace rund::node::accel::detail
