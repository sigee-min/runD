#pragma once

#include <accel/graph/factory/primitive/segmented/reduce/node.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode AccelSegmentedReduce(
    const AccelGraphBufferRef *const refs, const std::uint64_t ref_count,
    const rund::kernel::ReduceOp op, const rund::kernel::ReduceElement element,
    const std::uint64_t element_count,
    const std::uint64_t block_size) noexcept {
  return AccelSegmentedReduce(refs, ref_count,
                              rund::kernel::SegmentedReduceDesc{
                                  .op = op,
                                  .element = element,
                                  .element_count = element_count,
                                  .block_size = block_size,
                              });
}

} // namespace rund
