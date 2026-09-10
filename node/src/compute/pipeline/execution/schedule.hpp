#pragma once

#include "attempt.hpp"
#include "window.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <rund/storage.hpp>

namespace rund::compute::detail {

using PipelineResidencyScheduleFinalCompletion =
    void (*)(void *, residency::execution::ScheduleEvidence &&) noexcept;

struct PipelineExecutionSchedulePrepared final {
  node::accel::detail::PreparedResidencyScheduleRequest request{};
  std::array<std::shared_ptr<PipelineState>,
             node::accel::detail::ResidencyScheduleRoleCapacity>
      pipelines{};
  node::accel::detail::BackendResidencySchedulePreparation lowering{};
  std::shared_ptr<storage::Reservation> memory{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return request.plan_identity != 0u && request.epoch_count > 4u &&
           request.role_count ==
               node::accel::detail::ResidencyScheduleRoleCapacity &&
           lowering.check.ok && lowering.callbacks_async &&
           lowering.kind != node::accel::detail::
                                BackendResidencyScheduleLowering::Unsupported &&
           memory != nullptr && memory->committed();
  }
};

// One whole-run Compute terminal owner. The four slots are logical
// (bank,publication-parity) roles, not epochs; an epoch authenticates its slot
// with e mod 4 before state can be reused.
struct PipelineResidencyScheduleControl final {
  std::mutex gate{};
  node::accel::detail::PreparedResidencyScheduleControl native{};
  const residency::execution::Plan *plan{};
  residency::ExecutionLease lease{};
  std::array<std::shared_ptr<PipelineState>,
             node::accel::detail::ResidencyScheduleRoleCapacity>
      pipelines{};
  std::array<PipelineExecutionAttempt,
             node::accel::detail::ResidencyScheduleRoleCapacity>
      attempts{};
  PipelineResidencyWindowReleaseCompletion release{};
  PipelineResidencyScheduleFinalCompletion final{};
  void *user{};
  node::accel::detail::BackendResidencySchedulePreparation lowering{};
  bool active{};
};

[[nodiscard]] Status submit_pipeline_execution_schedule(
    const residency::execution::Plan &,
    const PipelineExecutionSchedulePrepared &,
    const residency::ExecutionLease &, PipelineResidencyWindowReleaseCompletion,
    PipelineResidencyScheduleFinalCompletion, void *,
    PipelineResidencyScheduleControl &,
    node::accel::detail::PreparedResidencyStreamControl &) noexcept;

[[nodiscard]] Status prepare_pipeline_execution_schedule(
    const residency::execution::Plan &,
    const std::array<std::shared_ptr<PipelineState>,
                     residency::execution::BankCapacity> &,
    PipelineExecutionSchedulePrepared &) noexcept;

[[nodiscard]] Status
signal_pipeline_execution_schedule(PipelineResidencyScheduleControl &,
                                   std::uint64_t epoch,
                                   Status admission) noexcept;

[[nodiscard]] Status
abort_pipeline_execution_schedule(PipelineResidencyScheduleControl &,
                                  Status failure) noexcept;

} // namespace rund::compute::detail
