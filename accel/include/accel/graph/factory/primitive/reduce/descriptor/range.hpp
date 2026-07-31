#pragma once

#include <accel/graph/factory/primitive/reduce/descriptor/full.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelReduce(const AccelGraphBufferRef *const refs,
            const std::uint64_t ref_count, const rund::kernel::ReduceOp op,
            const rund::kernel::ReduceElement element,
            const std::uint64_t element_count) noexcept {
  return AccelReduce(refs, ref_count, op, element, element_count,
                     element_count);
}

} // namespace rund
