#pragma once

#include <accel/graph/factory/primitive/compact/node.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode AccelCompact(
    const AccelGraphBufferRef *const refs, const std::uint64_t ref_count,
    const std::uint64_t element_count, const std::uint64_t output_capacity,
    const std::uint32_t flag_bytes, const std::uint32_t output_bytes) noexcept {
  return AccelCompact(refs, ref_count,
                      rund::kernel::CompactDesc{
                          .element_count = element_count,
                          .output_capacity = output_capacity,
                          .flag_bytes = flag_bytes,
                          .output_bytes = output_bytes,
                      });
}

} // namespace rund
