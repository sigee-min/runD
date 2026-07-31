#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/node.hpp>

#include "../bindings.hpp"

#include "../local.hpp"

namespace rund::node::accel::detail {

bool CompactBindingsOk(const rund::AccelGraphNode &node,
                       const rund::kernel::CompactPlan &plan) noexcept {
  if (node.buffer_count != 2u || node.buffers == nullptr) {
    return false;
  }
  const rund::AccelGraphBufferRef &flags = node.buffers[0];
  const rund::AccelGraphBufferRef &output = node.buffers[1];
  if (!AccelGraphBufferShapeValid(flags) ||
      !AccelGraphBufferShapeValid(output) ||
      flags.role != rund::kernel::BufferRole::Read ||
      output.role != rund::kernel::BufferRole::Write) {
    return false;
  }
  return AccelGraphBufferShape(flags).scalar_width_bytes == plan.flag_bytes &&
         AccelGraphBufferShape(output).scalar_width_bytes ==
             plan.output_bytes &&
         plan.flag_bytes == sizeof(rund::kernel::u32) &&
         plan.output_bytes == sizeof(rund::kernel::u32) &&
         AccelGraphBufferShape(flags).count == plan.element_count &&
         AccelGraphBufferShape(output).count == plan.output_capacity &&
         CollectiveUsageOk(AccelGraphBufferShape(flags).usage, false) &&
         CollectiveUsageOk(AccelGraphBufferShape(output).usage, true);
}

} // namespace rund::node::accel::detail
