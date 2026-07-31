#pragma once

#include "../kernel/bindings/histogram.hpp"
#include "../primitive/shape.hpp"

#include <kernel/program/compute/histogram/plan.hpp>

namespace rund::node::accel::detail {

[[nodiscard]] bool HistogramShapeOk(const rund::kernel::HistogramDesc &desc,
                                    const rund::kernel::HistogramPlan &plan,
                                    const HistogramBinds &bindings) noexcept;

} // namespace rund::node::accel::detail
