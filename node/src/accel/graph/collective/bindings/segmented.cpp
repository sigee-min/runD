#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/node.hpp>

#include "../bindings.hpp"

#include "../local.hpp"

namespace rund::node::accel::detail {

bool SegmentedScanBindingsOk(
    const rund::AccelGraphNode &node,
    const rund::kernel::SegmentedScanPlan &plan) noexcept {
  if (node.buffer_count != 3u || node.buffers == nullptr) {
    return false;
  }
  const rund::AccelGraphBufferRef &input = node.buffers[0];
  const rund::AccelGraphBufferRef &heads = node.buffers[1];
  const rund::AccelGraphBufferRef &output = node.buffers[2];
  if (!AccelGraphBufferShapeValid(input) ||
      !AccelGraphBufferShapeValid(heads) ||
      !AccelGraphBufferShapeValid(output) ||
      input.role != rund::kernel::BufferRole::Read ||
      heads.role != rund::kernel::BufferRole::Read ||
      output.role != rund::kernel::BufferRole::Write) {
    return false;
  }
  return AccelGraphBufferShape(input).scalar_width_bytes ==
             plan.element_bytes &&
         AccelGraphBufferShape(heads).scalar_width_bytes == plan.head_bytes &&
         AccelGraphBufferShape(output).scalar_width_bytes ==
             plan.element_bytes &&
         AccelGraphBufferShape(input).count == plan.element_count &&
         AccelGraphBufferShape(heads).count == plan.element_count &&
         AccelGraphBufferShape(output).count == plan.element_count &&
         CollectiveUsageOk(AccelGraphBufferShape(input).usage, false) &&
         CollectiveUsageOk(AccelGraphBufferShape(heads).usage, false) &&
         CollectiveUsageOk(AccelGraphBufferShape(output).usage, true);
}

} // namespace rund::node::accel::detail
