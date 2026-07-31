#pragma once

#include <accel/graph/factory/scan/basic/node.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelScan(const AccelGraphBufferRef *const refs, const std::uint64_t ref_count,
          const rund::kernel::ScanOp op,
          const rund::kernel::ScanElement element,
          const std::uint64_t element_count,
          const std::uint64_t block_size) noexcept {
  return AccelScan(refs, ref_count,
                   rund::kernel::ScanDesc{
                       .op = op,
                       .element = element,
                       .element_count = element_count,
                       .block_size = block_size,
                   });
}

} // namespace rund
