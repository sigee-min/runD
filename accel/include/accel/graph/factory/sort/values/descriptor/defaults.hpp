#pragma once

#include <accel/graph/factory/sort/values/descriptor/range.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelSort(const AccelGraphBufferRef *const refs, const std::uint64_t ref_count,
          const std::uint64_t element_count) noexcept {
  return AccelSort(refs, ref_count, rund::kernel::SortKey::U32,
                   rund::kernel::SortValue::U32, element_count);
}

} // namespace rund
