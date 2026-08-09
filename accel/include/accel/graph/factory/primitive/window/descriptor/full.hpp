#pragma once

#include <accel/graph/factory/primitive/window/node.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode AccelWindow(
    const AccelGraphBufferRef *const refs, const std::uint64_t ref_count,
    const kernel::WindowOp op, const kernel::WindowElement element,
    const kernel::WindowBoundary boundary, const kernel::ComputeDomain domain,
    const kernel::ComputeFixedFormat fixed_format,
    const std::uint64_t input_count, const std::uint64_t output_count,
    const std::uint64_t window_size, const std::uint64_t stride,
    const std::uint64_t pad_left) noexcept {
  return AccelWindow(refs, ref_count,
                     kernel::WindowDesc{
                         .op = op,
                         .element = element,
                         .boundary = boundary,
                         .domain = domain,
                         .fixed_format = fixed_format,
                         .input_count = input_count,
                         .output_count = output_count,
                         .window_size = window_size,
                         .stride = stride,
                         .pad_left = pad_left,
                     });
}

} // namespace rund
