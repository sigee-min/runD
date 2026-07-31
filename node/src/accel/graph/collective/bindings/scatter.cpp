#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/node.hpp>

#include "../bindings.hpp"

#include "../local.hpp"

namespace rund::node::accel::detail {

bool ScatterBindingsOk(const rund::AccelGraphNode &node,
                       const rund::kernel::ScatterPlan &plan) noexcept {
  if (node.buffer_count != 3u || node.buffers == nullptr) {
    return false;
  }
  const rund::AccelGraphBufferRef &values = node.buffers[0];
  const rund::AccelGraphBufferRef &indices = node.buffers[1];
  const rund::AccelGraphBufferRef &output = node.buffers[2];
  if (!AccelGraphBufferShapeValid(values) ||
      !AccelGraphBufferShapeValid(indices) ||
      !AccelGraphBufferShapeValid(output) ||
      values.role != rund::kernel::BufferRole::Read ||
      indices.role != rund::kernel::BufferRole::Read ||
      output.role != rund::kernel::BufferRole::Write) {
    return false;
  }
  return AccelGraphBufferShape(values).scalar_width_bytes ==
             plan.element_bytes &&
         AccelGraphBufferShape(indices).scalar_width_bytes ==
             plan.index_bytes &&
         AccelGraphBufferShape(output).scalar_width_bytes ==
             plan.element_bytes &&
         AccelGraphBufferShape(values).count == plan.element_count &&
         AccelGraphBufferShape(indices).count == plan.element_count &&
         AccelGraphBufferShape(output).count == plan.output_count &&
         CollectiveUsageOk(AccelGraphBufferShape(values).usage, false) &&
         CollectiveUsageOk(AccelGraphBufferShape(indices).usage, false) &&
         CollectiveUsageOk(AccelGraphBufferShape(output).usage, true);
}

bool ScatterReduceBindingsOk(
    const rund::AccelGraphNode &node,
    const rund::kernel::ScatterReducePlan &plan) noexcept {
  const bool bounded =
      plan.count_source != rund::kernel::ComputeCountSource::Descriptor;
  const std::uint64_t expected = bounded ? 4u : 3u;
  if (node.buffer_count != expected || node.buffers == nullptr) return false;
  const auto &values = node.buffers[0u];
  const auto &indices = node.buffers[1u];
  const auto &output = node.buffers[expected - 1u];
  if (!AccelGraphBufferShapeValid(values) ||
      !AccelGraphBufferShapeValid(indices) ||
      !AccelGraphBufferShapeValid(output) ||
      values.role != rund::kernel::BufferRole::Read ||
      indices.role != rund::kernel::BufferRole::Read ||
      output.role != rund::kernel::BufferRole::Write ||
      AccelGraphBufferShape(values).scalar_width_bytes != plan.element_bytes ||
      AccelGraphBufferShape(values).count != plan.element_count ||
      AccelGraphBufferShape(indices).scalar_width_bytes != plan.index_bytes ||
      AccelGraphBufferShape(indices).count != plan.element_count ||
      AccelGraphBufferShape(output).scalar_width_bytes != plan.element_bytes ||
      AccelGraphBufferShape(output).count != plan.output_count ||
      !CollectiveUsageOk(AccelGraphBufferShape(values).usage, false) ||
      !CollectiveUsageOk(AccelGraphBufferShape(indices).usage, false) ||
      !CollectiveUsageOk(AccelGraphBufferShape(output).usage, true)) return false;
  if (!bounded) return true;
  const auto &count = node.buffers[2u];
  return AccelGraphBufferShapeValid(count) &&
         count.role == rund::kernel::BufferRole::Read &&
         AccelGraphBufferShape(count).scalar_width_bytes ==
             rund::kernel::ComputeCountBytes(plan.count_source) &&
         AccelGraphBufferShape(count).count == 1u &&
         CollectiveUsageOk(AccelGraphBufferShape(count).usage, false);
}

} // namespace rund::node::accel::detail
