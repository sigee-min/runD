#include <accel/kernel/run.hpp>
#include <accel/kernel/run/binding.hpp>

#include "local.hpp"

#include <kernel/core/checked.hpp>

namespace rund::node::accel::detail {

bool AccumulateLatestProducerRoundtrip(
    const KernelExecution &execution, const ScheduledStepOrder &step_order,
    const rund::AccelRun &run, const std::size_t consumer_order,
    const std::uint64_t read_index, const rund::AccelRunBinding &read_binding,
    const std::uint64_t read_bytes,
    ProducerConsumerRoundtrip &result) noexcept {
  for (std::size_t producer_order = consumer_order; producer_order > 0u;
       --producer_order) {
    const KernelExecutionStep &producer =
        execution.steps[step_order.at(producer_order - 1u)];
    if (!producer.graph_binding_indices_ok) {
      return false;
    }
    for (std::uint64_t write_local = 0u;
         write_local < producer.graph_binding_indices.size(); ++write_local) {
      const std::uint64_t write_index =
          producer.graph_binding_indices[write_local];
      if (!BindingRoleIs(execution, write_index, run.binding_count,
                         rund::kernel::BufferRole::Write)) {
        continue;
      }
      const rund::AccelRunBinding &write_binding = run.bindings[write_index];
      if (write_binding.buffer == nullptr || read_binding.buffer == nullptr ||
          !SameBinding(write_binding, read_binding)) {
        continue;
      }

      std::uint64_t write_bytes = 0u;
      bool read_internal = false;
      bool write_internal = false;
      if (!BindingSpanBytes(write_binding, write_bytes) ||
          !BindingVisibilityIsInternal(execution, read_index, run.binding_count,
                                       read_internal) ||
          !BindingVisibilityIsInternal(execution, write_index,
                                       run.binding_count, write_internal)) {
        return false;
      }
      std::uint64_t *bucket = read_internal && write_internal
                                  ? &result.internal_bytes
                                  : &result.external_bytes;
      return rund::kernel::checked::add(*bucket, write_bytes, *bucket) &&
             rund::kernel::checked::add(*bucket, read_bytes, *bucket);
    }
  }
  return true;
}

} // namespace rund::node::accel::detail
