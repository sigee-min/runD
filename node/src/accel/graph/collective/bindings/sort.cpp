#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/node.hpp>

#include "../bindings.hpp"

#include "../local.hpp"

#include <kernel/program/compute/model.hpp>

namespace rund::node::accel::detail {

bool SortBindingsOk(const rund::AccelGraphNode &node,
                    const rund::kernel::SortPlan &plan) noexcept {
  const bool identity_values =
      plan.value == rund::kernel::SortValue::IdentityU32;
  const bool bounded =
      plan.count_source != rund::kernel::ComputeCountSource::Descriptor;
  const std::size_t reads = (identity_values ? 1u : 2u) + (bounded ? 1u : 0u);
  if (node.buffer_count != reads + 2u || node.buffers == nullptr) {
    return false;
  }
  const rund::AccelGraphBufferRef &read_keys = node.buffers[0];
  const rund::AccelGraphBufferRef &write_keys = node.buffers[reads];
  const rund::AccelGraphBufferRef &write_values = node.buffers[reads + 1u];
  if (!AccelGraphBufferShapeValid(read_keys) ||
      !AccelGraphBufferShapeValid(write_keys) ||
      !AccelGraphBufferShapeValid(write_values) ||
      read_keys.role != rund::kernel::BufferRole::Read ||
      write_keys.role != rund::kernel::BufferRole::Write ||
      write_values.role != rund::kernel::BufferRole::Write) {
    return false;
  }
  if (!identity_values) {
    const rund::AccelGraphBufferRef &read_values =
        node.buffers[bounded ? 2u : 1u];
    if (!AccelGraphBufferShapeValid(read_values) ||
        read_values.role != rund::kernel::BufferRole::Read ||
        AccelGraphBufferShape(read_values).scalar_width_bytes !=
            plan.value_bytes ||
        AccelGraphBufferShape(read_values).count != plan.element_count ||
        !CollectiveUsageOk(AccelGraphBufferShape(read_values).usage, false)) {
      return false;
    }
  }
  if (bounded) {
    const rund::AccelGraphBufferRef &count = node.buffers[1u];
    if (!AccelGraphBufferShapeValid(count) ||
        count.role != rund::kernel::BufferRole::Read ||
        AccelGraphBufferShape(count).scalar_width_bytes !=
            rund::kernel::ComputeCountBytes(plan.count_source) ||
        AccelGraphBufferShape(count).count != 1u ||
        !CollectiveUsageOk(AccelGraphBufferShape(count).usage, false)) {
      return false;
    }
  }
  return SortKeyWidthOk(AccelGraphBufferShape(read_keys).scalar_width_bytes) &&
         AccelGraphBufferShape(read_keys).scalar_width_bytes ==
             AccelGraphBufferShape(write_keys).scalar_width_bytes &&
         AccelGraphBufferShape(read_keys).scalar_width_bytes ==
             plan.key_bytes &&
         AccelGraphBufferShape(write_values).scalar_width_bytes ==
             plan.value_bytes &&
         (plan.value == rund::kernel::SortValue::U32 ||
          plan.value == rund::kernel::SortValue::IdentityU32) &&
         plan.value_bytes == sizeof(rund::kernel::u32) &&
         AccelGraphBufferShape(read_keys).count == plan.element_count &&
         AccelGraphBufferShape(write_keys).count == plan.element_count &&
         AccelGraphBufferShape(write_values).count == plan.element_count &&
         CollectiveUsageOk(AccelGraphBufferShape(read_keys).usage, false) &&
         CollectiveUsageOk(AccelGraphBufferShape(write_keys).usage, true) &&
         CollectiveUsageOk(AccelGraphBufferShape(write_values).usage, true);
}

} // namespace rund::node::accel::detail
