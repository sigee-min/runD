#include "local.hpp"

#include <algorithm>
#include <limits>

namespace rund::kernel {
namespace {

void AttachPartitionWorkMetadata(KernelProgramPlacementMetadata& metadata,
                                 const ScheduleView schedule,
                                 const std::span<const u32> ordered_packets,
                                 const std::span<const u64> resolved_work_units,
                                 const std::span<const u64> partition_work_units) {
  if (schedule.partitions == nullptr || schedule.partition_count == 0u) {
    return;
  }

  u64 min_work = std::numeric_limits<u64>::max();
  u64 max_work = 0u;
  u64 total_work = 0u;
  if (partition_work_units.size() == static_cast<std::size_t>(schedule.partition_count)) {
    for (const u64 partition_work : partition_work_units) {
      min_work = std::min(min_work, partition_work);
      max_work = std::max(max_work, partition_work);
      total_work = rund::math32::detail::ScalarSatAdd(total_work, partition_work);
    }
    metadata.min_partition_work_units = min_work == std::numeric_limits<u64>::max() ? 0u : min_work;
    metadata.max_partition_work_units = max_work;
    const u64 mean_work = total_work / schedule.partition_count;
    metadata.work_imbalance_milli =
        mean_work == 0u || max_work <= mean_work
            ? 0u
            : rund::math32::detail::ScalarSatMilliRatio(max_work - mean_work, mean_work);
    return;
  }

  if (resolved_work_units.size() != static_cast<std::size_t>(schedule.packet_count)) {
    return;
  }

  for (u32 partition_index = 0u; partition_index < schedule.partition_count; ++partition_index) {
    const Partition& partition = schedule.partitions[partition_index];
    u64 partition_work = 0u;
    for (u32 offset = 0u; offset < partition.size(); ++offset) {
      const u32 ordered_index = partition.begin + offset;
      const u32 packet = ordered_packets.empty() ? ordered_index : ordered_packets[ordered_index];
      partition_work = rund::math32::detail::ScalarSatAdd(
          partition_work,
          resolved_work_units[packet]);
    }
    min_work = std::min(min_work, partition_work);
    max_work = std::max(max_work, partition_work);
    total_work = rund::math32::detail::ScalarSatAdd(total_work, partition_work);
  }
  metadata.min_partition_work_units = min_work == std::numeric_limits<u64>::max() ? 0u : min_work;
  metadata.max_partition_work_units = max_work;
  const u64 mean_work = total_work / schedule.partition_count;
  metadata.work_imbalance_milli =
      mean_work == 0u || max_work <= mean_work
          ? 0u
          : rund::math32::detail::ScalarSatMilliRatio(max_work - mean_work, mean_work);
}

u32 CapacityImbalanceMilli(const ScheduleCompileRequest& request) {
  if (request.worker_capacity_milli.empty()) {
    return 0u;
  }
  u32 min_capacity = std::numeric_limits<u32>::max();
  u32 max_capacity = 0u;
  for (const u32 capacity : request.worker_capacity_milli) {
    if (capacity == 0u) {
      continue;
    }
    min_capacity = std::min(min_capacity, capacity);
    max_capacity = std::max(max_capacity, capacity);
  }
  return min_capacity == std::numeric_limits<u32>::max()
             ? 0u
             : rund::math32::detail::ScalarSatMilliRatio(max_capacity, min_capacity);
}

bool CurrentScheduleMetadataValid(const ScheduleCompileRequest& request,
                                  const ScheduleView schedule) {
  if (schedule.packet_count != request.packet_count ||
      schedule.execution_width != request.execution_width ||
      schedule.partitions == nullptr ||
      schedule.partition_count == 0u) {
    return request.packet_count == 0u && schedule.packet_count == 0u;
  }
  if (schedule.ordered_packet_count != 0u &&
      schedule.ordered_packet_count != schedule.packet_count) {
    return false;
  }
  return true;
}

} // namespace

KernelProgramPlacementMetadata BuildKernelProgramPlacementMetadata(
    const ScheduleCompileRequest& request,
    const ScheduleView schedule,
    const std::span<const u64> resolved_work_units,
    const std::span<const u64> partition_work_units,
    const WorkerBackendCapabilities backend_capabilities) {
  if (!CurrentScheduleMetadataValid(request, schedule)) {
    return KernelProgramPlacementMetadata{};
  }
  const std::span<const u32> ordered_packets(schedule.ordered_packet_indices,
                                             schedule.ordered_packet_count);
  KernelProgramPlacementMetadata metadata{
      .has_packet_hints = workspace_detail::HasPacketHints(request),
      .has_packet_work_units = request.packet_work_units.size() == static_cast<std::size_t>(request.packet_count),
      .alignment_group_packets =
          std::max<u32>(1u,
                        std::max({request.preferred_alignment_packets,
                                  request.locality_bucket_packets,
                                  request.alignment_group_packets})),
      .locality_bucket_crossing_count =
          workspace_placement::CountLocalityBucketCrossings(request, ordered_packets),
      .has_worker_capacity =
          request.worker_capacity_milli.size() >= static_cast<std::size_t>(schedule.useful_width),
      .worker_capacity_truth = request.trust_worker_capacity,
      .worker_capacity_imbalance_milli = CapacityImbalanceMilli(request),
      .affinity_truth_level = backend_capabilities.affinity_truth_level,
      .affinity_hint_only = backend_capabilities.affinity_truth_level == WorkerTruthLevel::HintOnly,
      .affinity_placement_reason = workspace_placement::AffinityPlacementReason(backend_capabilities),
  };
  AttachPartitionWorkMetadata(metadata,
                              schedule,
                              ordered_packets,
                              resolved_work_units,
                              partition_work_units);
  return metadata;
}

KernelProgramPlacementMetadata BuildKernelProgramPlacementMetadata(
    const ScheduleCompileRequest& request,
    const ScheduleView schedule,
    const std::span<const u64> resolved_work_units,
    const WorkerBackendCapabilities backend_capabilities) {
  return BuildKernelProgramPlacementMetadata(request,
                                            schedule,
                                            resolved_work_units,
                                            {},
                                            backend_capabilities);
}

KernelProgramPlacementMetadata BuildKernelProgramPlacementMetadata(
    const ScheduleCompileRequest& request,
    const ScheduleView schedule,
    const WorkerBackendCapabilities backend_capabilities) {
  return BuildKernelProgramPlacementMetadata(request, schedule, {}, backend_capabilities);
}

} // namespace rund::kernel
