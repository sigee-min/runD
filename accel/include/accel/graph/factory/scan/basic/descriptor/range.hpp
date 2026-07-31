#pragma once

#include <accel/graph/factory/scan/basic/descriptor/full.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelScan(const AccelGraphBufferRef *const refs, const std::uint64_t ref_count,
          const rund::kernel::ScanOp op,
          const rund::kernel::ScanElement element,
          const std::uint64_t element_count) noexcept {
  return AccelScan(refs, ref_count, op, element, element_count, element_count);
}

} // namespace rund
