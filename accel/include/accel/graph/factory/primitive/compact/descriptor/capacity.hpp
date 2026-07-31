#pragma once

#include <accel/graph/factory/primitive/compact/descriptor/full.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelCompact(const AccelGraphBufferRef *const refs,
             const std::uint64_t ref_count, const std::uint64_t element_count,
             const std::uint64_t output_capacity) noexcept {
  return AccelCompact(refs, ref_count, element_count, output_capacity,
                      sizeof(rund::kernel::u32), sizeof(rund::kernel::u32));
}

[[nodiscard]] inline AccelGraphNode
AccelCompact(const AccelGraphBufferRef *const refs,
             const std::uint64_t ref_count,
             const std::uint64_t element_count) noexcept {
  return AccelCompact(refs, ref_count, element_count, element_count);
}

} // namespace rund
