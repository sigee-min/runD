#pragma once

#include "../../pipeline.hpp"
#include "../../run.hpp"

#include "../../../backend/pipeline/failure.hpp"
#include "../../evidence.hpp"
#include "../../model.hpp"
#include "../backend.hpp"
#include "../demand.hpp"
#include "../expand.hpp"
#include "../recurrence.hpp"
#include "../registry.hpp"
#include "../reservation.hpp"
#include "../structure.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace rund::node::accel::detail {

// This draft is one cold-call workspace. It owns no retained registry,
// template, or backend state; the PipelineState and route vectors become
// reachable only after every phase has succeeded and the caller publishes the
// returned PreparedKernelPipeline.
struct PipelineMaterializationDraft final {
  PreparedPipelineFailureContext failure{};
  PreparedPipelineFailure failure_override{};
  PreparedKernelPipelineReservation reservation{};
  std::array<BackendTemplateRouteDemand, PreparedPipelineStepCapacity>
      template_route_demands{};
  std::shared_ptr<prepared::PipelineState> pipeline{};
  std::vector<BackendBatchEntry> batch_templates{};
  ExpandedPipeline expanded{};
  PreparedPipelineMemory backend_memory{};
  rund::AccelRunFacts preparation{};
  bool has_failure_override{};
};

[[nodiscard]] PreparedKernelPipeline
reject_pipeline(const PreparedPipelineFailureContext &failure,
                const char *reason) noexcept;

[[nodiscard]] bool validate_pipeline_request(
    std::span<const PreparedKernelRun *const> runs,
    std::span<const std::uint8_t> barriers,
    std::span<const std::uint32_t> declared_steps,
    std::span<const BackendRecurrence> recurrences,
    std::span<const BackendPublish> publications) noexcept;

[[nodiscard]] const char *admit_pipeline_resources(
    const rund::AccelContext &context,
    std::span<const PreparedKernelRun *const> runs,
    std::span<const BackendRecurrence> recurrences,
    std::span<const BackendPublish> publications,
    std::uint32_t declared_step_count, std::uint32_t generation_stride,
    bool profile_steps, PreparedKernelTemplateRegistry *templates,
    PipelineMaterializationDraft &draft,
    PipelineBudgetTransaction &budget_transaction) noexcept;

[[nodiscard]] const char *
materialize_pipeline_routes(const rund::AccelContext &context,
                            std::span<const PreparedKernelRun *const> runs,
                            std::span<const BackendRecurrence> recurrences,
                            PreparedKernelTemplateRegistry &registry,
                            PipelineMaterializationDraft &draft) noexcept;

[[nodiscard]] PreparedKernelPipeline finish_pipeline_materialization(
    std::span<const std::uint8_t> barriers,
    std::span<const std::uint32_t> declared_steps,
    std::span<const BackendPublish> publications,
    std::uint32_t declared_step_count, std::uint32_t generation_stride,
    bool profile_steps, PreparedKernelTemplateRegistry &registry,
    PipelineBudgetTransaction &budget_transaction,
    PipelineMaterializationDraft &draft) noexcept;

} // namespace rund::node::accel::detail
