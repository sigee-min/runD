#pragma once

#include <accel/graph/factory/primitive/partition/node.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelPartition(const AccelGraphBufferRef *const refs,
               const std::uint64_t ref_count, const std::uint64_t element_count,
               const std::uint32_t flag_bytes,
               const std::uint32_t value_bytes) noexcept {
  return AccelPartition(refs, ref_count,
                        rund::kernel::PartitionDesc{
                            .element_count = element_count,
                            .flag_bytes = flag_bytes,
                            .value_bytes = value_bytes,
                        });
}

} // namespace rund
