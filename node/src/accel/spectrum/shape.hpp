#pragma once

#include <kernel/program/compute/spectrum/plan.hpp>

#include "../kernel/bindings/spectrum.hpp"
#include "../primitive/shape.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] bool SpectrumShapeOk(const rund::kernel::SpectrumDesc &desc,
                                   const rund::kernel::SpectrumPlan &plan,
                                   const SpectrumBinds &bindings) noexcept;

} // namespace rund::node::accel::detail
