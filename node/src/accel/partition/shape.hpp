#pragma once

#include "../kernel/bindings/partition.hpp"
#include "../primitive/shape.hpp"

#include <kernel/program/compute/partition/plan.hpp>

namespace rund::node::accel::detail {

[[nodiscard]] bool PartitionShapeOk(const rund::kernel::PartitionDesc &desc,
                                    const rund::kernel::PartitionPlan &plan,
                                    const PartitionBinds &bindings) noexcept;

} // namespace rund::node::accel::detail
