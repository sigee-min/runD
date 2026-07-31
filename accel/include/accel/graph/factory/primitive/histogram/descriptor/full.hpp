#pragma once

#include <accel/graph/factory/primitive/histogram/node.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode AccelHistogram(
    const AccelGraphBufferRef *const refs, const std::uint64_t ref_count,
    const rund::kernel::HistogramIndex index,
    const rund::kernel::HistogramCount count, const std::uint64_t element_count,
    const std::uint64_t bin_count) noexcept {
  return AccelHistogram(refs, ref_count,
                        rund::kernel::HistogramDesc{
                            .index = index,
                            .count = count,
                            .element_count = element_count,
                            .bin_count = bin_count,
                        });
}

} // namespace rund
