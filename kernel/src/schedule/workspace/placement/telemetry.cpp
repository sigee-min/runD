#include "local.hpp"

namespace rund::kernel {

void AttachKernelProgramPlacementTelemetry(Telemetry& telemetry,
                                           const KernelProgramPlacementMetadata& metadata) {
  telemetry.max_partition_work_units = metadata.max_partition_work_units;
  telemetry.min_partition_work_units = metadata.min_partition_work_units;
  telemetry.work_imbalance_milli = metadata.work_imbalance_milli;
  telemetry.work_imbalance_measured = metadata.has_packet_work_units ||
                                      metadata.max_partition_work_units != 0u ||
                                      metadata.min_partition_work_units != 0u;
  telemetry.locality_bucket_crossing_count = metadata.locality_bucket_crossing_count;
  telemetry.locality_bucket_crossing_measured = metadata.has_packet_hints;
  telemetry.has_worker_capacity = metadata.has_worker_capacity;
  telemetry.worker_capacity_truth = metadata.worker_capacity_truth;
  telemetry.worker_capacity_imbalance_milli = metadata.worker_capacity_imbalance_milli;
  telemetry.affinity_truth_level = metadata.affinity_truth_level;
  telemetry.affinity_hint_only = metadata.affinity_hint_only;
}

} // namespace rund::kernel
