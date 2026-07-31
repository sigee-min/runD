#pragma once

#include <accel/graph/factory/primitive/stencil/descriptor/full.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelStencil(const AccelGraphBufferRef *const refs,
             const std::uint64_t ref_count,
             const rund::kernel::StencilElement element,
             const std::uint64_t element_count,
             const std::uint64_t radius = 1u) noexcept {
  return AccelStencil(refs, ref_count, rund::kernel::StencilOp::Sum, element,
                      rund::kernel::StencilBoundary::Clamp, element_count,
                      radius);
}

} // namespace rund
