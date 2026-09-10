#include "../pipeline.hpp"
#include "../run.hpp"
#include "materialize/internal.hpp"

namespace rund::node::accel::detail {

PreparedKernelPipeline
PrepareKernelPipeline(const rund::AccelContext &context,
                      const std::span<const PreparedKernelRun *const> runs,
                      const std::span<const std::uint8_t> barriers,
                      const std::span<const std::uint32_t> declared_steps,
                      const std::span<const BackendRecurrence> recurrences,
                      const std::span<const BackendPublish> publications,
                      const std::uint32_t declared_step_count,
                      const std::uint32_t generation_stride,
                      const bool profile_steps,
                      PreparedKernelTemplateRegistry *const templates) {
  const rund::AccelCheck invalid{false, "accel_kernel_run_invalid"};
  PipelineMaterializationDraft draft{};
  draft.failure.stage(PreparedPipelineFailureStage::CommonValidation);
  if (!validate_pipeline_request(runs, barriers, declared_steps, recurrences,
                                 publications)) {
    return reject_pipeline(draft.failure, invalid.reason);
  }

  PipelineBudgetTransaction budget_transaction{};
  if (const char *const reason = admit_pipeline_resources(
          context, runs, recurrences, publications, declared_step_count,
          generation_stride, profile_steps, templates, draft,
          budget_transaction);
      reason != nullptr) {
    return reject_pipeline(draft.failure, reason);
  }
  if (templates == nullptr) {
    return reject_pipeline(draft.failure, "accel_kernel_template_invalid");
  }
  const char *const materialize_reason = materialize_pipeline_routes(
      context, runs, recurrences, *templates, draft);
  if (materialize_reason != nullptr) {
    if (draft.has_failure_override) {
      return PreparedKernelPipeline{.failure = draft.failure_override};
    }
    return reject_pipeline(draft.failure, materialize_reason);
  }
  return finish_pipeline_materialization(
      barriers, declared_steps, publications, declared_step_count,
      generation_stride, profile_steps, *templates, budget_transaction, draft);
}

} // namespace rund::node::accel::detail
