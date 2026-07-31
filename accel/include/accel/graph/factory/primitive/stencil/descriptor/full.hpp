#pragma once

#include <accel/graph/factory/primitive/stencil/node.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelStencil(const AccelGraphBufferRef *const refs,
             const std::uint64_t ref_count, const rund::kernel::StencilOp op,
             const rund::kernel::StencilElement element,
             const rund::kernel::StencilBoundary boundary,
             const std::uint64_t element_count,
             const std::uint64_t radius = 1u) noexcept {
  return AccelStencil(refs, ref_count,
                      rund::kernel::StencilDesc{
                          .op = op,
                          .element = element,
                          .boundary = boundary,
                          .element_count = element_count,
                          .radius = radius,
                      });
}

} // namespace rund
