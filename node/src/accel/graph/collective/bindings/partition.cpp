#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/node.hpp>

#include "../bindings.hpp"

#include "../local.hpp"

namespace rund::node::accel::detail {

bool PartitionBindingsOk(const rund::AccelGraphNode &node,
                         const rund::kernel::PartitionPlan &plan) noexcept {
  if (node.buffer_count != 3u || node.buffers == nullptr) {
    return false;
  }
  const rund::AccelGraphBufferRef &flags = node.buffers[0];
  const rund::AccelGraphBufferRef &values = node.buffers[1];
  const rund::AccelGraphBufferRef &output = node.buffers[2];
  if (!AccelGraphBufferShapeValid(flags) ||
      !AccelGraphBufferShapeValid(values) ||
      !AccelGraphBufferShapeValid(output) ||
      flags.role != rund::kernel::BufferRole::Read ||
      values.role != rund::kernel::BufferRole::Read ||
      output.role != rund::kernel::BufferRole::Write) {
    return false;
  }
  return AccelGraphBufferShape(flags).scalar_width_bytes == plan.flag_bytes &&
         AccelGraphBufferShape(values).scalar_width_bytes == plan.value_bytes &&
         AccelGraphBufferShape(output).scalar_width_bytes == plan.value_bytes &&
         AccelGraphBufferShape(flags).count == plan.element_count &&
         AccelGraphBufferShape(values).count == plan.element_count &&
         AccelGraphBufferShape(output).count == plan.element_count &&
         CollectiveUsageOk(AccelGraphBufferShape(flags).usage, false) &&
         CollectiveUsageOk(AccelGraphBufferShape(values).usage, false) &&
         CollectiveUsageOk(AccelGraphBufferShape(output).usage, true);
}

} // namespace rund::node::accel::detail
