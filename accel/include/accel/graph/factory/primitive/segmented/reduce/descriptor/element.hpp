#pragma once

#include <accel/graph/factory/primitive/segmented/reduce/descriptor/range.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelSegmentedReduce(const AccelGraphBufferRef *const refs,
                     const std::uint64_t ref_count,
                     const rund::kernel::ReduceOp op,
                     const std::uint64_t element_count) noexcept {
  return AccelSegmentedReduce(refs, ref_count, op,
                              rund::kernel::ReduceElement::U32, element_count);
}

} // namespace rund
