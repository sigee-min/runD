#pragma once

#include <accel/graph/factory/scan/segmented/descriptor/range.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelSegmentedScan(const AccelGraphBufferRef *const refs,
                   const std::uint64_t ref_count,
                   const rund::kernel::SegmentedScanOp op,
                   const std::uint64_t element_count) noexcept {
  return AccelSegmentedScan(refs, ref_count, op,
                            rund::kernel::SegmentedScanElement::U32,
                            element_count);
}

} // namespace rund
