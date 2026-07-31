#pragma once

#include "../kernel/bindings/compact.hpp"
#include "../primitive/shape.hpp"

#include <kernel/program/compute/compact/plan.hpp>

namespace rund::node::accel::detail {

[[nodiscard]] bool CompactShapeOk(const rund::kernel::CompactDesc &desc,
                                  const rund::kernel::CompactPlan &plan,
                                  const CompactBinds &bindings) noexcept;

} // namespace rund::node::accel::detail
