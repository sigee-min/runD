#pragma once

#include <accel/graph/factory/primitive/gather/descriptor/full.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelGather(const AccelGraphBufferRef *const refs,
            const std::uint64_t ref_count, const std::uint64_t element_count,
            const std::uint64_t source_count) noexcept {
  return AccelGather(refs, ref_count, rund::kernel::GatherElement::U32,
                     element_count, source_count);
}

} // namespace rund
