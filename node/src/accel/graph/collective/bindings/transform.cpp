#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/node.hpp>

#include "../bindings.hpp"

#include "../local.hpp"

namespace rund::node::accel::detail {

bool TransformBindingsOk(const rund::AccelGraphNode &node,
                         const rund::kernel::TransformPlan &plan) noexcept {
  if (!plan.ok || plan.layout != rund::kernel::TransformLayout::Split ||
      node.buffer_count != 4u || node.buffers == nullptr) {
    return false;
  }
  const rund::AccelGraphBufferRef &input_real = node.buffers[0];
  const rund::AccelGraphBufferRef &input_imag = node.buffers[1];
  const rund::AccelGraphBufferRef &output_real = node.buffers[2];
  const rund::AccelGraphBufferRef &output_imag = node.buffers[3];
  if (!AccelGraphBufferShapeValid(input_real) ||
      !AccelGraphBufferShapeValid(input_imag) ||
      !AccelGraphBufferShapeValid(output_real) ||
      !AccelGraphBufferShapeValid(output_imag) ||
      input_real.role != rund::kernel::BufferRole::Read ||
      input_imag.role != rund::kernel::BufferRole::Read ||
      output_real.role != rund::kernel::BufferRole::Write ||
      output_imag.role != rund::kernel::BufferRole::Write) {
    return false;
  }
  const std::uint64_t bytes =
      AccelGraphBufferShape(input_real).scalar_width_bytes;
  return (bytes == 4u || bytes == 8u) &&
         AccelGraphBufferShape(input_imag).scalar_width_bytes == bytes &&
         AccelGraphBufferShape(output_real).scalar_width_bytes == bytes &&
         AccelGraphBufferShape(output_imag).scalar_width_bytes == bytes &&
         AccelGraphBufferShape(input_real).count == plan.element_count &&
         AccelGraphBufferShape(input_imag).count == plan.element_count &&
         AccelGraphBufferShape(output_real).count == plan.element_count &&
         AccelGraphBufferShape(output_imag).count == plan.element_count &&
         CollectiveUsageOk(AccelGraphBufferShape(input_real).usage, false) &&
         CollectiveUsageOk(AccelGraphBufferShape(input_imag).usage, false) &&
         CollectiveUsageOk(AccelGraphBufferShape(output_real).usage, true) &&
         CollectiveUsageOk(AccelGraphBufferShape(output_imag).usage, true);
}

} // namespace rund::node::accel::detail
