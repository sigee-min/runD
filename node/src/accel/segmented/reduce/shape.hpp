#pragma once

#include "../../kernel/bindings/segmented.hpp"

#include <kernel/program/compute/segmented/reduce/model.hpp>

namespace rund::node::accel::detail {

[[nodiscard]] bool SegmentedReduceShapeOk(
    const rund::kernel::SegmentedReduceDesc &desc,
    const rund::kernel::SegmentedReducePlan &plan,
    const SegmentedReduceBinds &bindings) noexcept;

} // namespace rund::node::accel::detail
