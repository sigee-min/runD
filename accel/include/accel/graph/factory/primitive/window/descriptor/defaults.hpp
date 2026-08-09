#pragma once

#include <accel/graph/factory/primitive/window/descriptor/full.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelWindow(const AccelGraphBufferRef *const refs,
            const std::uint64_t ref_count, const std::uint64_t input_count,
            const std::uint64_t output_count, const std::uint64_t window_size,
            const std::uint64_t stride = 1u,
            const std::uint64_t pad_left = 0u) noexcept {
  return AccelWindow(refs, ref_count, kernel::WindowOp::Sum,
                     kernel::WindowElement::U32, kernel::WindowBoundary::Clamp,
                     kernel::ComputeDomain::U32, {}, input_count, output_count,
                     window_size, stride, pad_left);
}

} // namespace rund
