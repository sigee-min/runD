#pragma once

#include <accel/graph/factory/node/base.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelMap(const rund::kernel::ComputeIR &ir,
         const AccelGraphBufferRef *const refs, const std::uint64_t ref_count,
         const std::uint64_t element_count) {
  return accel_graph_factory_detail::MapNode(ir, refs, ref_count,
                                             element_count);
}

} // namespace rund
