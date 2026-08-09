#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/node.hpp>

#include "../bindings.hpp"
#include "../local.hpp"

namespace rund::node::accel::detail {

bool WindowBindingsOk(const rund::AccelGraphNode &node,
                      const rund::kernel::WindowPlan &plan) noexcept {
  if (node.buffer_count != 2u || node.buffers == nullptr) {
    return false;
  }
  const rund::AccelGraphBufferRef &input = node.buffers[0u];
  const rund::AccelGraphBufferRef &output = node.buffers[1u];
  if (!AccelGraphBufferShapeValid(input) ||
      !AccelGraphBufferShapeValid(output) ||
      input.role != rund::kernel::BufferRole::Read ||
      output.role != rund::kernel::BufferRole::Write) {
    return false;
  }
  return AccelGraphBufferShape(input).scalar_width_bytes ==
             plan.element_bytes &&
         AccelGraphBufferShape(output).scalar_width_bytes ==
             plan.element_bytes &&
         AccelGraphBufferShape(input).count == plan.input_count &&
         AccelGraphBufferShape(output).count == plan.output_count &&
         CollectiveUsageOk(AccelGraphBufferShape(input).usage, false) &&
         CollectiveUsageOk(AccelGraphBufferShape(output).usage, true);
}

} // namespace rund::node::accel::detail
