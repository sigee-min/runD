#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/node.hpp>

#include "../bindings.hpp"

#include "../local.hpp"

namespace rund::node::accel::detail {

bool GatherBindingsOk(const rund::AccelGraphNode &node,
                      const rund::kernel::GatherPlan &plan) noexcept {
  const bool bounded = plan.count_source !=
                       rund::kernel::ComputeCountSource::Descriptor;
  if (node.buffer_count != (bounded ? 4u : 3u) || node.buffers == nullptr) {
    return false;
  }
  const rund::AccelGraphBufferRef &values = node.buffers[0];
  const rund::AccelGraphBufferRef &indices = node.buffers[1];
  const rund::AccelGraphBufferRef *const logical_count =
      bounded ? &node.buffers[2] : nullptr;
  const rund::AccelGraphBufferRef &output = node.buffers[bounded ? 3u : 2u];
  if (!AccelGraphBufferShapeValid(values) ||
      !AccelGraphBufferShapeValid(indices) ||
      !AccelGraphBufferShapeValid(output) ||
      (bounded && !AccelGraphBufferShapeValid(*logical_count)) ||
      values.role != rund::kernel::BufferRole::Read ||
      indices.role != rund::kernel::BufferRole::Read ||
      (bounded && logical_count->role != rund::kernel::BufferRole::Read) ||
      output.role != rund::kernel::BufferRole::Write) {
    return false;
  }
  return AccelGraphBufferShape(values).scalar_width_bytes ==
             plan.element_bytes &&
         AccelGraphBufferShape(indices).scalar_width_bytes ==
             plan.index_bytes &&
         AccelGraphBufferShape(output).scalar_width_bytes ==
             plan.element_bytes &&
         (!bounded ||
          (AccelGraphBufferShape(*logical_count).scalar_width_bytes ==
               rund::kernel::ComputeCountBytes(plan.count_source) &&
           AccelGraphBufferShape(*logical_count).count == 1u)) &&
         AccelGraphBufferShape(values).count >= plan.source_count &&
         AccelGraphBufferShape(indices).count == plan.element_count &&
         AccelGraphBufferShape(output).count == plan.element_count &&
         CollectiveUsageOk(AccelGraphBufferShape(values).usage, false) &&
         CollectiveUsageOk(AccelGraphBufferShape(indices).usage, false) &&
         (!bounded ||
          CollectiveUsageOk(AccelGraphBufferShape(*logical_count).usage,
                            false)) &&
         CollectiveUsageOk(AccelGraphBufferShape(output).usage, true);
}

} // namespace rund::node::accel::detail
