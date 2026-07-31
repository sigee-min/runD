#pragma once

#include <accel/graph/factory/scan/basic/descriptor/element.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelScan(const AccelGraphBufferRef *const refs, const std::uint64_t ref_count,
          const std::uint64_t element_count) noexcept {
  return AccelScan(refs, ref_count, rund::kernel::ScanOp::ExclusiveSum,
                   element_count);
}

} // namespace rund
