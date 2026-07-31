#include "../local.hpp"

namespace rund::kernel {

KernelProgramCapacityProof BuildKernelProgramCapacityProof(const Workspace& workspace,
                                                           const WorkspaceReservation& required,
                                                           const bool checked,
                                                           const char* reason) {
  const WorkspaceCapacity available = GetWorkspaceCapacity(workspace);
  const bool satisfied = workspace_detail::ReservationSatisfiedByCapacity(available, required);
  return KernelProgramCapacityProof{
      .checked = checked,
      .satisfied = satisfied,
      .reason = checked ? (satisfied ? "pass" : reason) : "not_checked",
      .required = ToKernelProgramCapacitySet(required),
      .available = ToKernelProgramCapacitySet(available),
      .no_alloc_capacity_margin = workspace_detail::MinimumCapacityMargin(required, available),
  };
}

void AttachKernelProgramCapacityTelemetry(Telemetry& telemetry,
                                          const KernelProgramCapacityProof& proof) {
  telemetry.capacity_checked = proof.checked;
  telemetry.capacity_satisfied = proof.satisfied;
  telemetry.required_schedule_partition_capacity = proof.required.schedule_partition_capacity;
  telemetry.available_schedule_partition_capacity = proof.available.schedule_partition_capacity;
  telemetry.required_packet_capacity = proof.required.packet_capacity;
  telemetry.available_packet_capacity = proof.available.packet_capacity;
  telemetry.required_fold_slot_capacity = proof.required.fold_slot_capacity;
  telemetry.available_fold_slot_capacity = proof.available.fold_slot_capacity;
  telemetry.required_worker_stats_capacity = proof.required.worker_stats_capacity;
  telemetry.available_worker_stats_capacity = proof.available.worker_stats_capacity;
  telemetry.no_alloc_capacity_margin = proof.no_alloc_capacity_margin;
  telemetry.no_alloc_capacity_margin_measured = proof.checked;
}

} // namespace rund::kernel
