#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/node.hpp>

#include "../bindings.hpp"

#include "../local.hpp"

namespace rund::node::accel::detail {

bool SpectrumBindingsOk(const rund::AccelGraphNode &node,
                        const rund::kernel::SpectrumPlan &plan) noexcept {
  if (!plan.ok || node.buffers == nullptr) {
    return false;
  }
  const bool has_vectors = plan.vector_count != 0u;
  if (node.buffer_count != (has_vectors ? 4u : 3u)) {
    return false;
  }

  const rund::AccelGraphBufferRef &input = node.buffers[0u];
  const rund::AccelGraphBufferRef &values = node.buffers[1u];
  const rund::AccelGraphBufferRef &status = node.buffers[has_vectors ? 3u : 2u];
  if (!AccelGraphBufferShapeValid(input) ||
      !AccelGraphBufferShapeValid(values) ||
      !AccelGraphBufferShapeValid(status) ||
      input.role != rund::kernel::BufferRole::Read ||
      values.role != rund::kernel::BufferRole::Write ||
      status.role != rund::kernel::BufferRole::Write ||
      AccelGraphBufferShape(input).scalar_width_bytes != plan.element_bytes ||
      AccelGraphBufferShape(values).scalar_width_bytes != plan.element_bytes ||
      AccelGraphBufferShape(status).scalar_width_bytes !=
          sizeof(rund::kernel::u32) ||
      AccelGraphBufferShape(input).count != plan.input_count ||
      AccelGraphBufferShape(values).count != plan.value_count ||
      AccelGraphBufferShape(status).count != plan.status_count ||
      !CollectiveUsageOk(AccelGraphBufferShape(input).usage, false) ||
      !CollectiveUsageOk(AccelGraphBufferShape(values).usage, true) ||
      !CollectiveUsageOk(AccelGraphBufferShape(status).usage, true)) {
    return false;
  }
  if (!has_vectors) {
    return true;
  }
  const rund::AccelGraphBufferRef &vectors = node.buffers[2u];
  return AccelGraphBufferShapeValid(vectors) &&
         vectors.role == rund::kernel::BufferRole::Write &&
         AccelGraphBufferShape(vectors).scalar_width_bytes ==
             plan.element_bytes &&
         AccelGraphBufferShape(vectors).count == plan.vector_count &&
         CollectiveUsageOk(AccelGraphBufferShape(vectors).usage, true);
}

} // namespace rund::node::accel::detail
