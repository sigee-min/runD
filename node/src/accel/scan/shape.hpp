#pragma once

#include "../kernel/bindings/scan.hpp"

#include <kernel/program/compute/scan/model.hpp>

namespace rund::node::accel::detail {

[[nodiscard]] bool ScanShapeOk(const rund::kernel::ScanDesc &desc,
                               const rund::kernel::ScanPlan &plan) noexcept;

[[nodiscard]] bool ScanResidentShapeOk(const rund::kernel::ScanPlan &plan,
                                       const ScanBinds &bindings) noexcept;

} // namespace rund::node::accel::detail
