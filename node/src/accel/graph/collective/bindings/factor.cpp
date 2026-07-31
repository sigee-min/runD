#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/node.hpp>

#include "../bindings.hpp"

#include "../local.hpp"

namespace rund::node::accel::detail {

bool FactorBindingsOk(const rund::AccelGraphNode &node,
                      const rund::kernel::FactorPlan &plan) noexcept {
  if (!plan.ok || node.buffers == nullptr) {
    return false;
  }
  const bool needs_aux = plan.op == rund::kernel::FactorOp::LU;
  const std::uint64_t expected = needs_aux ? 4u : 3u;
  if (node.buffer_count != expected) {
    return false;
  }

  const rund::AccelGraphBufferRef &input = node.buffers[0u];
  const rund::AccelGraphBufferRef &factor = node.buffers[1u];
  const rund::AccelGraphBufferRef &status = node.buffers[needs_aux ? 3u : 2u];
  if (!AccelGraphBufferShapeValid(input) ||
      !AccelGraphBufferShapeValid(factor) ||
      !AccelGraphBufferShapeValid(status) ||
      input.role != rund::kernel::BufferRole::Read ||
      factor.role != rund::kernel::BufferRole::Write ||
      status.role != rund::kernel::BufferRole::Write ||
      AccelGraphBufferShape(input).scalar_width_bytes != plan.element_bytes ||
      AccelGraphBufferShape(factor).scalar_width_bytes != plan.element_bytes ||
      AccelGraphBufferShape(status).scalar_width_bytes !=
          sizeof(rund::kernel::u32) ||
      AccelGraphBufferShape(input).count != plan.input_count ||
      AccelGraphBufferShape(factor).count != plan.factor_count ||
      AccelGraphBufferShape(status).count != plan.status_count ||
      !CollectiveUsageOk(AccelGraphBufferShape(input).usage, false) ||
      !CollectiveUsageOk(AccelGraphBufferShape(factor).usage, true) ||
      !CollectiveUsageOk(AccelGraphBufferShape(status).usage, true)) {
    return false;
  }
  if (!needs_aux) {
    return true;
  }
  const rund::AccelGraphBufferRef &aux = node.buffers[2u];
  return AccelGraphBufferShapeValid(aux) &&
         aux.role == rund::kernel::BufferRole::Write &&
         AccelGraphBufferShape(aux).scalar_width_bytes ==
             sizeof(rund::kernel::u32) &&
         AccelGraphBufferShape(aux).count == plan.aux_count &&
         CollectiveUsageOk(AccelGraphBufferShape(aux).usage, true);
}

} // namespace rund::node::accel::detail
