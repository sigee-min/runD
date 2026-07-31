#pragma once

#include <accel/graph/factory/sort/values/descriptor/full.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelSort(const AccelGraphBufferRef *const refs, const std::uint64_t ref_count,
          const rund::kernel::SortKey key, const rund::kernel::SortValue value,
          const std::uint64_t element_count) noexcept {
  return AccelSort(refs, ref_count, key, value, element_count, 0u);
}

} // namespace rund
