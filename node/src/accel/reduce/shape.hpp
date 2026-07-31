#pragma once

#include "../kernel/bindings/reduce.hpp"
#include "../primitive/shape.hpp"

#include <kernel/program/compute/model.hpp>
#include <kernel/program/compute/reduce/model.hpp>
#include <kernel/program/compute/reduce/plan.hpp>

namespace rund::node::accel::detail {

[[nodiscard]] bool ReduceShapeOk(const rund::kernel::ReduceDesc &desc,
                                 const rund::kernel::ReducePlan &plan,
                                 const ReduceBinds &bindings) noexcept;

} // namespace rund::node::accel::detail
