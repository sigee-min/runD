#pragma once

#include "model.hpp"

namespace rund::measure::compute::preparation_memory {

template <class T>
inline void CaptureFailure(PreparationMemoryObservation &observed,
                           const ::rund::compute::Result<T> &failure) noexcept {
  observed.code = failure.code();
  observed.reason = failure.reason();
  observed.location = failure.location();
  observed.error = failure.error();
}

[[nodiscard]] bool PlanContract(const ::rund::compute::PipelinePlan &plan,
                                Backend backend) noexcept;
[[nodiscard]] bool PreciseFailure(
    const PreparationMemoryObservation &observed) noexcept;
[[nodiscard]] constexpr bool
NoAllocation(const ::rund::compute::MemoryCounter before,
             const ::rund::compute::MemoryCounter after) noexcept {
  return after.current <= before.current && after.peak == before.peak &&
         after.cumulative == before.cumulative &&
         after.reused == before.reused && after.budget == before.budget;
}
[[nodiscard]] constexpr bool
NoAllocation(const ::rund::compute::MemoryStats &before,
             const ::rund::compute::MemoryStats &after) noexcept {
  return before.backend == after.backend && before.scope == after.scope &&
         NoAllocation(before.host, after.host) &&
         NoAllocation(before.frame, after.frame) &&
         NoAllocation(before.tile, after.tile) &&
         NoAllocation(before.resident, after.resident) &&
         NoAllocation(before.staging, after.staging) &&
         NoAllocation(before.device, after.device) &&
         NoAllocation(before.transfer, after.transfer);
}
[[nodiscard]] bool
PlanObservationContract(const PreparationMemoryObservation &observed) noexcept;
[[nodiscard]] bool
PreparedMemoryContract(const ::rund::compute::PipelinePlan &plan,
                       const ::rund::compute::MemoryStats &memory,
                       Backend backend) noexcept;
[[nodiscard]] bool CaptureLargestRetainedGroup(
    PreparationMemoryObservation &observed,
    const ::rund::compute::Pipeline &pipeline) noexcept;
[[nodiscard]] bool CaptureBackendPreparation(
    PreparationMemoryObservation &observed,
    const ::rund::compute::Pipeline &pipeline, Backend backend) noexcept;

} // namespace rund::measure::compute::preparation_memory
