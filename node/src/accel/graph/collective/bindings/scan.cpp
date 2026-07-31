#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/node.hpp>

#include "../bindings.hpp"

#include "../local.hpp"

#include <kernel/program/compute/model.hpp>

namespace rund::node::accel::detail {

bool ScanBindingsOk(const rund::AccelGraphNode &node,
                    const rund::kernel::ScanPlan &plan) noexcept {
  const bool bounded =
      plan.count_source != rund::kernel::ComputeCountSource::Descriptor;
  if (node.buffer_count != (bounded ? 3u : 2u) || node.buffers == nullptr) {
    return false;
  }
  const rund::AccelGraphBufferRef &input = node.buffers[0];
  const rund::AccelGraphBufferRef &output = node.buffers[bounded ? 2u : 1u];
  if (!AccelGraphBufferShapeValid(input) ||
      !AccelGraphBufferShapeValid(output) ||
      input.role != rund::kernel::BufferRole::Read ||
      output.role != rund::kernel::BufferRole::Write) {
    return false;
  }
  if (bounded) {
    const rund::AccelGraphBufferRef &count = node.buffers[1u];
    if (!AccelGraphBufferShapeValid(count) ||
        count.role != rund::kernel::BufferRole::Read ||
        AccelGraphBufferShape(count).count != 1u ||
        AccelGraphBufferShape(count).scalar_width_bytes !=
            rund::kernel::ComputeCountBytes(plan.count_source) ||
        !CollectiveUsageOk(AccelGraphBufferShape(count).usage, false)) {
      return false;
    }
  }
  return AccelGraphBufferShape(input).scalar_width_bytes ==
             plan.element_bytes &&
         AccelGraphBufferShape(output).scalar_width_bytes ==
             plan.element_bytes &&
         AccelGraphBufferShape(input).count == plan.element_count &&
         AccelGraphBufferShape(output).count == plan.element_count &&
         CollectiveUsageOk(AccelGraphBufferShape(input).usage, false) &&
         CollectiveUsageOk(AccelGraphBufferShape(output).usage, true);
}

} // namespace rund::node::accel::detail
