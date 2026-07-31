#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/node.hpp>

#include "../bindings.hpp"

#include "../local.hpp"

namespace rund::node::accel::detail {

bool SolveBindingsOk(const rund::AccelGraphNode &node,
                     const rund::kernel::SolvePlan &plan) noexcept {
  if (!plan.ok || node.buffers == nullptr) {
    return false;
  }
  const bool factor_input = plan.input == rund::kernel::SolveInput::Factor;
  const bool needs_aux =
      factor_input && plan.factor == rund::kernel::FactorOp::LU;
  const std::uint64_t expected = factor_input ? (needs_aux ? 5u : 4u) : 4u;
  if (node.buffer_count != expected) {
    return false;
  }

  const rund::AccelGraphBufferRef &primary = node.buffers[0u];
  const std::uint64_t rhs_index = factor_input ? (needs_aux ? 2u : 1u) : 1u;
  const std::uint64_t output_index = rhs_index + 1u;
  const std::uint64_t status_index = output_index + 1u;
  const rund::AccelGraphBufferRef &rhs = node.buffers[rhs_index];
  const rund::AccelGraphBufferRef &output = node.buffers[output_index];
  const rund::AccelGraphBufferRef &status = node.buffers[status_index];
  const std::uint64_t primary_count =
      factor_input ? plan.factor_count : plan.matrix_count;
  if (!AccelGraphBufferShapeValid(primary) ||
      !AccelGraphBufferShapeValid(rhs) || !AccelGraphBufferShapeValid(output) ||
      !AccelGraphBufferShapeValid(status) ||
      primary.role != rund::kernel::BufferRole::Read ||
      rhs.role != rund::kernel::BufferRole::Read ||
      output.role != rund::kernel::BufferRole::Write ||
      status.role != rund::kernel::BufferRole::Write ||
      AccelGraphBufferShape(primary).scalar_width_bytes != plan.element_bytes ||
      AccelGraphBufferShape(rhs).scalar_width_bytes != plan.element_bytes ||
      AccelGraphBufferShape(output).scalar_width_bytes != plan.element_bytes ||
      AccelGraphBufferShape(status).scalar_width_bytes !=
          sizeof(rund::kernel::u32) ||
      AccelGraphBufferShape(primary).count != primary_count ||
      AccelGraphBufferShape(rhs).count != plan.rhs_count ||
      AccelGraphBufferShape(output).count != plan.output_count ||
      AccelGraphBufferShape(status).count != plan.status_count ||
      !CollectiveUsageOk(AccelGraphBufferShape(primary).usage, false) ||
      !CollectiveUsageOk(AccelGraphBufferShape(rhs).usage, false) ||
      !CollectiveUsageOk(AccelGraphBufferShape(output).usage, true) ||
      !CollectiveUsageOk(AccelGraphBufferShape(status).usage, true)) {
    return false;
  }
  if (!needs_aux) {
    return true;
  }
  const rund::AccelGraphBufferRef &aux = node.buffers[1u];
  return AccelGraphBufferShapeValid(aux) &&
         aux.role == rund::kernel::BufferRole::Read &&
         AccelGraphBufferShape(aux).scalar_width_bytes ==
             sizeof(rund::kernel::u32) &&
         AccelGraphBufferShape(aux).count == plan.aux_count &&
         CollectiveUsageOk(AccelGraphBufferShape(aux).usage, false);
}

} // namespace rund::node::accel::detail
