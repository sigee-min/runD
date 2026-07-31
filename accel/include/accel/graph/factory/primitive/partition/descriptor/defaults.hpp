#pragma once

#include <accel/graph/factory/primitive/partition/descriptor/full.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelPartition(const AccelGraphBufferRef *const refs,
               const std::uint64_t ref_count,
               const std::uint64_t element_count) noexcept {
  return AccelPartition(refs, ref_count, element_count,
                        sizeof(rund::kernel::u32), sizeof(rund::kernel::u32));
}

} // namespace rund
