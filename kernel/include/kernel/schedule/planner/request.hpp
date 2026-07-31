#pragma once

#include <kernel/schedule/planner/policy.hpp>

#include <span>

namespace rund::kernel {

struct PacketPlacementHint {
  u32 locality_bucket_id = 0u;
  u32 cache_line_group = 0u;
  u32 preferred_contiguous_span = 1u;
  u64 work_units = 1u;
};

struct PartitionRequest {
  u32 packet_count = 0;
  u32 execution_width = 1;
  PartitionIntent intent = PartitionIntent::StaticWidth;
  u32 preferred_alignment_packets = 1;
  std::span<const u64> packet_work_units{};
  std::span<const u32> worker_capacity_milli{};
  bool trust_worker_capacity = false;
};

struct ScheduleCompileRequest {
  u32 packet_count = 0;
  u32 execution_width = 1;
  PartitionIntent intent = PartitionIntent::StaticWidth;
  PlacementPolicy placement = PlacementPolicy::Uniform;
  AllocationPolicy allocation = AllocationPolicy::AllowGrowth;
  u32 preferred_alignment_packets = 1;
  u32 locality_bucket_packets = 1;
  u32 alignment_group_packets = 1;
  std::span<const u64> packet_work_units{};
  std::span<const u32> worker_capacity_milli{};
  std::span<const PacketPlacementHint> packet_hints{};
  bool trust_worker_capacity = false;
};

PartitionRequest ToPartitionRequest(const ScheduleCompileRequest& request);
ScheduleCompileRequest ToScheduleCompileRequest(const PartitionRequest& request);

} // namespace rund::kernel
