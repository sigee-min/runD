#include "local.hpp"

#include <utility>

namespace rund::kernel::dispatch::detail {

Result ExecuteSerial(const Plan& plan, Telemetry telemetry) {
  u32 executed_partitions = 0u;
  DispatchAdapter adapter{
      .context = plan.context,
      .dispatch = plan.dispatch,
      .ordered_packet_indices = plan.ordered_packet_indices,
      .ordered_packet_count = plan.ordered_packet_count,
  };
  try {
    for (u32 index = 0u; index < plan.partition_count; ++index) {
      InvokeMappedPartition(&adapter, plan.partitions[index]);
      executed_partitions += 1u;
    }
  } catch (...) {
    serial::ApplySingleWorkerStats(plan.worker_stats_sink,
                                   plan.require_no_allocation,
                                   executed_partitions,
                                   telemetry);
    return serial::BuildSerialResult(false, std::move(telemetry));
  }

  serial::ApplySingleWorkerStats(plan.worker_stats_sink,
                                 plan.require_no_allocation,
                                 executed_partitions,
                                 telemetry);
  return serial::BuildSerialResult(true, std::move(telemetry));
}

} // namespace rund::kernel::dispatch::detail
