#pragma once

#include <accel/graph/factory/scan/basic/descriptor/range.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelScan(const AccelGraphBufferRef *const refs, const std::uint64_t ref_count,
          const rund::kernel::ScanOp op,
          const std::uint64_t element_count) noexcept {
  return AccelScan(refs, ref_count, op, rund::kernel::ScanElement::U32,
                   element_count);
}

} // namespace rund
