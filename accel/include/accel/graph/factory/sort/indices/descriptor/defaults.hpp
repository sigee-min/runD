#pragma once

#include <accel/graph/factory/sort/indices/descriptor/range.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelArgsort(const AccelGraphBufferRef *const refs,
             const std::uint64_t ref_count,
             const std::uint64_t element_count) noexcept {
  return AccelArgsort(refs, ref_count, rund::kernel::SortKey::U32,
                      element_count);
}

} // namespace rund
