#pragma once

#include <accel/graph/factory/primitive/scatter/descriptor/full.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelScatter(const AccelGraphBufferRef *const refs,
             const std::uint64_t ref_count, const std::uint64_t element_count,
             const std::uint64_t output_count) noexcept {
  return AccelScatter(refs, ref_count, rund::kernel::ScatterElement::U32,
                      element_count, output_count);
}

} // namespace rund
