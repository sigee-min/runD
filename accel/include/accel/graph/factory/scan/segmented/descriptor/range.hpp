#pragma once

#include <accel/graph/factory/scan/segmented/descriptor/full.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelSegmentedScan(const AccelGraphBufferRef *const refs,
                   const std::uint64_t ref_count,
                   const rund::kernel::SegmentedScanOp op,
                   const rund::kernel::SegmentedScanElement element,
                   const std::uint64_t element_count) noexcept {
  return AccelSegmentedScan(refs, ref_count, op, element, element_count,
                            element_count);
}

} // namespace rund
