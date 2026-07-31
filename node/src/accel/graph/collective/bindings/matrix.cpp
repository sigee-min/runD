#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/node.hpp>

#include "../bindings.hpp"

#include "../local.hpp"

namespace rund::node::accel::detail {

bool MatrixBindingsOk(const rund::AccelGraphNode &node,
                      const rund::kernel::MatrixPlan &plan) noexcept {
  if (!plan.ok || node.buffers == nullptr) {
    return false;
  }
  const bool transpose = plan.op == rund::kernel::MatrixOp::Transpose;
  if (node.buffer_count != (transpose ? 2u : 3u)) {
    return false;
  }

  const rund::AccelGraphBufferRef &left = node.buffers[0];
  const rund::AccelGraphBufferRef &output = node.buffers[transpose ? 1u : 2u];
  if (!AccelGraphBufferShapeValid(left) ||
      !AccelGraphBufferShapeValid(output) ||
      left.role != rund::kernel::BufferRole::Read ||
      output.role != rund::kernel::BufferRole::Write ||
      AccelGraphBufferShape(left).scalar_width_bytes != plan.element_bytes ||
      AccelGraphBufferShape(output).scalar_width_bytes != plan.element_bytes ||
      AccelGraphBufferShape(left).count != plan.left_count ||
      AccelGraphBufferShape(output).count != plan.output_count ||
      !CollectiveUsageOk(AccelGraphBufferShape(left).usage, false) ||
      !CollectiveUsageOk(AccelGraphBufferShape(output).usage, true)) {
    return false;
  }
  if (transpose) {
    return true;
  }

  const rund::AccelGraphBufferRef &right = node.buffers[1];
  return AccelGraphBufferShapeValid(right) &&
         right.role == rund::kernel::BufferRole::Read &&
         AccelGraphBufferShape(right).scalar_width_bytes ==
             plan.element_bytes &&
         AccelGraphBufferShape(right).count == plan.right_count &&
         CollectiveUsageOk(AccelGraphBufferShape(right).usage, false);
}

} // namespace rund::node::accel::detail
