#include "../local.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <limits>
#include <type_traits>

namespace rund::kernel {
namespace {

template <class Value>
[[nodiscard]] bool AddStorageBytes(const u64 capacity, u64 &bytes) noexcept {
  static_assert(std::is_trivially_destructible_v<Value>);
  u64 extent = 0u;
  return checked::mul(capacity, static_cast<u64>(sizeof(Value)), extent) &&
         checked::add(bytes, extent, bytes);
}

template <class Value>
[[nodiscard]] bool AddStorageBytes(const std::vector<Value> &values,
                                   u64 &bytes) noexcept {
  static_assert(sizeof(std::size_t) <= sizeof(u64));
  return AddStorageBytes<Value>(static_cast<u64>(values.capacity()), bytes);
}

} // namespace

namespace workspace_detail {

WorkspaceRetentionPlan
PlanWorkspaceRetention(WorkspaceReservation reservation) noexcept {
  reservation = NormalizeReservation(reservation);
  u64 bytes = 0u;
  const bool valid =
      AddStorageBytes<Partition>(reservation.schedule_partition_capacity,
                                 bytes) &&
      AddStorageBytes<u64>(reservation.packet_work_unit_capacity, bytes) &&
      AddStorageBytes<u32>(reservation.ordered_packet_capacity, bytes) &&
      AddStorageBytes<u32>(reservation.packet_partition_capacity, bytes) &&
      AddStorageBytes<u32>(reservation.ordered_packet_scratch_capacity,
                           bytes) &&
      AddStorageBytes<u64>(reservation.partition_load_capacity, bytes) &&
      AddStorageBytes<u32>(reservation.partition_count_capacity, bytes) &&
      AddStorageBytes<u32>(reservation.partition_offset_capacity, bytes) &&
      AddStorageBytes<u32>(reservation.partition_write_offset_capacity,
                           bytes) &&
      AddStorageBytes<u64>(reservation.fold_slot_capacity, bytes) &&
      AddStorageBytes<u32>(reservation.fold_slot_capacity, bytes) &&
      AddStorageBytes<FoldGraphNode>(reservation.fold_graph_node_capacity,
                                     bytes) &&
      AddStorageBytes<FoldGraphEdge>(reservation.fold_graph_edge_capacity,
                                     bytes) &&
      AddStorageBytes<u32>(reservation.worker_stats_capacity, bytes) &&
      AddStorageBytes<u64>(reservation.worker_stats_capacity, bytes) &&
      AddStorageBytes<u64>(reservation.worker_stats_capacity, bytes) &&
      AddStorageBytes<u64>(reservation.worker_stats_capacity, bytes);
  return WorkspaceRetentionPlan{
      .reservation = reservation,
      .bytes = valid ? bytes : std::numeric_limits<u64>::max(),
      .ok = valid,
      .reason = valid ? "pass" : "workspace_retention_overflow",
  };
}

u64 WorkspaceRetainedBytes(const Workspace &workspace) noexcept {
  u64 bytes = 0u;
  const bool valid =
      AddStorageBytes(workspace.schedule.partitions, bytes) &&
      AddStorageBytes(workspace.packet_work_units, bytes) &&
      AddStorageBytes(workspace.ordered_packet_indices, bytes) &&
      AddStorageBytes(workspace.packet_partition_indices, bytes) &&
      AddStorageBytes(workspace.ordered_packet_scratch, bytes) &&
      AddStorageBytes(workspace.partition_loads, bytes) &&
      AddStorageBytes(workspace.partition_counts, bytes) &&
      AddStorageBytes(workspace.partition_offsets, bytes) &&
      AddStorageBytes(workspace.partition_write_offsets, bytes) &&
      AddStorageBytes(workspace.fold_slots.values, bytes) &&
      AddStorageBytes(workspace.fold_graph.partition_fold_slots, bytes) &&
      AddStorageBytes(workspace.fold_graph.nodes, bytes) &&
      AddStorageBytes(workspace.fold_graph.reduction_edges, bytes) &&
      AddStorageBytes(workspace.worker_stats_partitions_per_worker, bytes) &&
      AddStorageBytes(workspace.worker_stats_start_offset_ns, bytes) &&
      AddStorageBytes(workspace.worker_stats_elapsed_ns, bytes) &&
      AddStorageBytes(workspace.worker_stats_tail_wait_ns, bytes);
  return valid ? bytes : std::numeric_limits<u64>::max();
}

} // namespace workspace_detail

bool ReserveWorkspace(Workspace &workspace,
                      const WorkspaceReservation &reservation) {
  const workspace_detail::WorkspaceRetentionPlan planned =
      workspace_detail::PlanWorkspaceRetention(reservation);
  if (!planned.ok) {
    return false;
  }
  const WorkspaceReservation &normalized = planned.reservation;
  try {
    workspace.schedule.partitions.reserve(
        normalized.schedule_partition_capacity);
    workspace.packet_work_units.reserve(normalized.packet_work_unit_capacity);
    workspace.ordered_packet_indices.reserve(
        normalized.ordered_packet_capacity);
    workspace.packet_partition_indices.reserve(
        normalized.packet_partition_capacity);
    workspace.ordered_packet_scratch.reserve(
        normalized.ordered_packet_scratch_capacity);
    workspace.partition_loads.reserve(normalized.partition_load_capacity);
    workspace.partition_counts.reserve(normalized.partition_count_capacity);
    workspace.partition_offsets.reserve(normalized.partition_offset_capacity);
    workspace.partition_write_offsets.reserve(
        normalized.partition_write_offset_capacity);
    workspace.fold_slots.values.reserve(normalized.fold_slot_capacity);
    workspace.fold_graph.partition_fold_slots.reserve(
        normalized.fold_slot_capacity);
    workspace.fold_graph.nodes.reserve(normalized.fold_graph_node_capacity);
    workspace.fold_graph.reduction_edges.reserve(
        normalized.fold_graph_edge_capacity);
    workspace.worker_stats_partitions_per_worker.reserve(
        normalized.worker_stats_capacity);
    workspace.worker_stats_start_offset_ns.reserve(
        normalized.worker_stats_capacity);
    workspace.worker_stats_elapsed_ns.reserve(normalized.worker_stats_capacity);
    workspace.worker_stats_tail_wait_ns.reserve(
        normalized.worker_stats_capacity);
  } catch (...) {
    return false;
  }
  return true;
}

WorkspaceCapacity GetWorkspaceCapacity(const Workspace &workspace) {
  return WorkspaceCapacity{
      .schedule_partition_capacity = rund::math32::detail::ScalarSatU32(
          workspace.schedule.partitions.capacity()),
      .packet_capacity = rund::math32::detail::ScalarSatU32(
          std::min({workspace.packet_work_units.capacity(),
                    workspace.ordered_packet_indices.capacity(),
                    workspace.packet_partition_indices.capacity(),
                    workspace.ordered_packet_scratch.capacity()})),
      .packet_work_unit_capacity = rund::math32::detail::ScalarSatU32(
          workspace.packet_work_units.capacity()),
      .ordered_packet_capacity = rund::math32::detail::ScalarSatU32(
          workspace.ordered_packet_indices.capacity()),
      .packet_partition_capacity = rund::math32::detail::ScalarSatU32(
          workspace.packet_partition_indices.capacity()),
      .ordered_packet_scratch_capacity = rund::math32::detail::ScalarSatU32(
          workspace.ordered_packet_scratch.capacity()),
      .partition_load_capacity = rund::math32::detail::ScalarSatU32(
          workspace.partition_loads.capacity()),
      .partition_count_capacity = rund::math32::detail::ScalarSatU32(
          workspace.partition_counts.capacity()),
      .partition_offset_capacity = rund::math32::detail::ScalarSatU32(
          workspace.partition_offsets.capacity()),
      .partition_write_offset_capacity = rund::math32::detail::ScalarSatU32(
          workspace.partition_write_offsets.capacity()),
      .fold_slot_capacity = rund::math32::detail::ScalarSatU32(
          std::min(workspace.fold_slots.values.capacity(),
                   workspace.fold_graph.partition_fold_slots.capacity())),
      .fold_graph_node_capacity = rund::math32::detail::ScalarSatU32(
          workspace.fold_graph.nodes.capacity()),
      .fold_graph_edge_capacity = rund::math32::detail::ScalarSatU32(
          workspace.fold_graph.reduction_edges.capacity()),
      .worker_stats_capacity = rund::math32::detail::ScalarSatU32(
          std::min({workspace.worker_stats_partitions_per_worker.capacity(),
                    workspace.worker_stats_start_offset_ns.capacity(),
                    workspace.worker_stats_elapsed_ns.capacity(),
                    workspace.worker_stats_tail_wait_ns.capacity()})),
  };
}

bool WorkspaceSatisfiesReservation(const Workspace &workspace,
                                   const WorkspaceReservation &reservation) {
  return workspace_detail::ReservationSatisfiedByCapacity(
      GetWorkspaceCapacity(workspace), reservation);
}

} // namespace rund::kernel
