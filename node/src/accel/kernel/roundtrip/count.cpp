#include <accel/kernel/run.hpp>
#include <accel/kernel/run/binding.hpp>

#include "local.hpp"

#include <kernel/core/checked.hpp>

namespace rund::node::accel::detail {
ProducerConsumerRoundtrip
CountProducerConsumerRoundtripBytes(const KernelExecution &execution,
                                    const ScheduledStepOrder &step_order,
                                    const rund::AccelRun &run) noexcept {
  ProducerConsumerRoundtrip result{};
  for (std::size_t consumer_order = 0u; consumer_order < step_order.size();
       ++consumer_order) {
    const KernelExecutionStep &consumer =
        execution.steps[step_order.at(consumer_order)];
    if (!consumer.graph_binding_indices_ok) {
      return RejectRoundtrip();
    }
    for (std::uint64_t read_local = 0u;
         read_local < consumer.graph_binding_indices.size(); ++read_local) {
      const std::uint64_t read_index =
          consumer.graph_binding_indices[read_local];
      if (!BindingRoleIs(execution, read_index, run.binding_count,
                         rund::kernel::BufferRole::Read)) {
        continue;
      }
      const rund::AccelRunBinding &read_binding = run.bindings[read_index];
      std::uint64_t read_bytes = 0u;
      if (!BindingSpanBytes(read_binding, read_bytes) ||
          !AccumulateLatestProducerRoundtrip(
              execution, step_order, run, consumer_order, read_index,
              read_binding, read_bytes, result)) {
        return RejectRoundtrip();
      }
    }
  }
  if (!rund::kernel::checked::add(result.internal_bytes,
                                  result.external_bytes)) {
    return RejectRoundtrip();
  }
  return result;
}

} // namespace rund::node::accel::detail
