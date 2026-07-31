#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/node.hpp>

#include "../bindings.hpp"

#include "../local.hpp"

namespace rund::node::accel::detail {

bool HistogramBindingsOk(const rund::AccelGraphNode &node,
                         const rund::kernel::HistogramPlan &plan) noexcept {
  if (node.buffer_count != 2u || node.buffers == nullptr) {
    return false;
  }
  const rund::AccelGraphBufferRef &bins = node.buffers[0];
  const rund::AccelGraphBufferRef &counts = node.buffers[1];
  if (!AccelGraphBufferShapeValid(bins) ||
      !AccelGraphBufferShapeValid(counts) ||
      bins.role != rund::kernel::BufferRole::Read ||
      counts.role != rund::kernel::BufferRole::Write) {
    return false;
  }
  return AccelGraphBufferShape(bins).scalar_width_bytes == plan.index_bytes &&
         AccelGraphBufferShape(counts).scalar_width_bytes == plan.count_bytes &&
         AccelGraphBufferShape(bins).count == plan.element_count &&
         AccelGraphBufferShape(counts).count == plan.bin_count &&
         CollectiveUsageOk(AccelGraphBufferShape(bins).usage, false) &&
         CollectiveUsageOk(AccelGraphBufferShape(counts).usage, true);
}

} // namespace rund::node::accel::detail
