#pragma once

#include <accel/graph/factory/sort/indices/descriptor/full.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelArgsort(const AccelGraphBufferRef *const refs,
             const std::uint64_t ref_count, const rund::kernel::SortKey key,
             const std::uint64_t element_count) noexcept {
  return AccelArgsort(refs, ref_count, key, element_count, 0u);
}

} // namespace rund
