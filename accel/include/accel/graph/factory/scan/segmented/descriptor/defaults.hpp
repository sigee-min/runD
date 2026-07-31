#pragma once

#include <accel/graph/factory/scan/segmented/descriptor/element.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelSegmentedScan(const AccelGraphBufferRef *const refs,
                   const std::uint64_t ref_count,
                   const std::uint64_t element_count) noexcept {
  return AccelSegmentedScan(refs, ref_count,
                            rund::kernel::SegmentedScanOp::ExclusiveSum,
                            element_count);
}

} // namespace rund
