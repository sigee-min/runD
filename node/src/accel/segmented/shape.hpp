#pragma once

#include "../kernel/bindings/segmented.hpp"

#include <kernel/program/compute/segmented/scan/model.hpp>

namespace rund::node::accel::detail {

[[nodiscard]] bool SegmentedScanShapeOk(
    const rund::kernel::SegmentedScanDesc &desc,
    const rund::kernel::SegmentedScanPlan &plan,
    const SegmentedScanBinds &bindings) noexcept;

} // namespace rund::node::accel::detail
