#pragma once

#include <accel/graph/factory/primitive/histogram/descriptor/full.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelHistogram(const AccelGraphBufferRef *const refs,
               const std::uint64_t ref_count, const std::uint64_t element_count,
               const std::uint64_t bin_count) noexcept {
  return AccelHistogram(refs, ref_count, rund::kernel::HistogramIndex::U32,
                        rund::kernel::HistogramCount::U32, element_count,
                        bin_count);
}

} // namespace rund
