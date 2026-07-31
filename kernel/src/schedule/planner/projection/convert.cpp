#include "../local.hpp"

#include <algorithm>

namespace rund::kernel {

PartitionRequest ToPartitionRequest(const ScheduleCompileRequest& request) {
  return PartitionRequest{
      .packet_count = request.packet_count,
      .execution_width = request.execution_width,
      .intent = request.intent,
      .preferred_alignment_packets =
          std::max<u32>(1u,
                        std::max({request.preferred_alignment_packets,
                                  request.locality_bucket_packets,
                                  request.alignment_group_packets})),
      .packet_work_units = request.packet_work_units,
      .worker_capacity_milli = request.worker_capacity_milli,
      .trust_worker_capacity = request.trust_worker_capacity,
  };
}

ScheduleCompileRequest ToScheduleCompileRequest(const PartitionRequest& request) {
  return ScheduleCompileRequest{
      .packet_count = request.packet_count,
      .execution_width = request.execution_width,
      .intent = request.intent,
      .placement = PlacementPolicy::Uniform,
      .allocation = AllocationPolicy::AllowGrowth,
      .preferred_alignment_packets = request.preferred_alignment_packets,
      .locality_bucket_packets = 1u,
      .alignment_group_packets = request.preferred_alignment_packets,
      .packet_work_units = request.packet_work_units,
      .worker_capacity_milli = request.worker_capacity_milli,
      .trust_worker_capacity = request.trust_worker_capacity,
  };
}

} // namespace rund::kernel
