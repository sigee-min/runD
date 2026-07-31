#pragma once

#include <accel/graph/factory/primitive/stencil/descriptor/op.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelStencil(const AccelGraphBufferRef *const refs,
             const std::uint64_t ref_count,
             const std::uint64_t element_count) noexcept {
  return AccelStencil(refs, ref_count, rund::kernel::StencilOp::Sum,
                      element_count);
}

} // namespace rund
