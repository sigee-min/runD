#pragma once

#include <accel/graph/factory/scan/segmented/node.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelSegmentedScan(const AccelGraphBufferRef *const refs,
                   const std::uint64_t ref_count,
                   const rund::kernel::SegmentedScanOp op,
                   const rund::kernel::SegmentedScanElement element,
                   const std::uint64_t element_count,
                   const std::uint64_t block_size) noexcept {
  return AccelSegmentedScan(refs, ref_count,
                            rund::kernel::SegmentedScanDesc{
                                .op = op,
                                .element = element,
                                .element_count = element_count,
                                .block_size = block_size,
                            });
}

} // namespace rund
