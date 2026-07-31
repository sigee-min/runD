#pragma once

#include <accel/graph/factory/sort/values/node.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelArgsort(const AccelGraphBufferRef *const refs,
             const std::uint64_t ref_count, const rund::kernel::SortKey key,
             const std::uint64_t element_count,
             const std::uint32_t key_bits) noexcept {
  return AccelSort(refs, ref_count,
                   rund::kernel::SortDesc{
                       .key = key,
                       .value = rund::kernel::SortValue::IdentityU32,
                       .element_count = element_count,
                       .radix_bits = 8u,
                       .key_bits = key_bits,
                       .stable = true,
                   });
}

} // namespace rund
