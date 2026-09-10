#include "../../pipeline.hpp"
#include "internal.hpp"

#include <kernel/core/checked.hpp>
#include <rund/counter.hpp>

#include <new>
#include <utility>

namespace rund::node::accel::detail {

using ::rund::kernel::checked::add;
using ::rund::kernel::checked::mul;

namespace {

[[nodiscard]] const char *
account_pipeline(const std::span<const std::uint32_t> declared_steps,
                 const std::uint32_t declared_step_count,
                 const std::uint32_t generation_stride,
                 PipelineMaterializationDraft &draft) noexcept {
  prepared::PipelineState &pipeline = *draft.pipeline;
  pipeline.counts = {};
  if (draft.expanded.command_count == 0u ||
      !PreparePipelineStatusLayout(
          pipeline.status, declared_steps, declared_step_count,
          draft.expanded.command_count, generation_stride)) {
    return "accel_kernel_run_invalid";
  }
  if (draft.expanded.compact_aggregate) {
    if (draft.expanded.aggregates.size() != 1u) {
      return "accel_kernel_run_invalid";
    }
    const NestedAggregate &aggregate = draft.expanded.aggregates.front();
    for (std::size_t index = 0u; index < pipeline.size; ++index) {
      draft.failure.template_route(static_cast<std::uint32_t>(index));
      NestedTemplateRouteProjection route{};
      if (pipeline.states[index] == nullptr ||
          !aggregate.shape.project(index, route)) {
        return "accel_kernel_run_invalid";
      }
      prepared::Accumulate(pipeline.counts, *pipeline.states[index],
                           route.occurrence_count);
    }
    return nullptr;
  }
  for (const BackendBatchEntry &command : draft.expanded.commands) {
    draft.failure.occurrence_route(command);
    if (command.template_index >= pipeline.size ||
        pipeline.states[command.template_index] == nullptr) {
      return "accel_kernel_run_invalid";
    }
    if (command.transducer == NoTileTransducer) {
      prepared::Accumulate(pipeline.counts,
                           *pipeline.states[command.template_index]);
      continue;
    }
    if (command.transducer >= draft.expanded.transducers.size()) {
      return "accel_kernel_run_invalid";
    }
    const TileTransducer &transducer =
        draft.expanded.transducers[command.transducer];
    const std::uint64_t template_end =
        static_cast<std::uint64_t>(transducer.template_first) +
        transducer.template_count;
    if (transducer.template_count == 0u || template_end > pipeline.size) {
      return "accel_kernel_run_invalid";
    }
    // One physical transducer command represents the complete authored
    // Action subrange for this outer window. Evidence remains logical: count
    // every original template once while the backend reports the smaller
    // physical dispatch count independently.
    for (std::uint32_t offset = 0u; offset < transducer.template_count;
         ++offset) {
      const std::size_t template_index = transducer.template_first + offset;
      draft.failure.template_route(static_cast<std::uint32_t>(template_index));
      if (pipeline.states[template_index] == nullptr) {
        return "accel_kernel_run_invalid";
      }
      prepared::Accumulate(pipeline.counts, *pipeline.states[template_index]);
    }
  }
  return nullptr;
}

} // namespace

PreparedKernelPipeline finish_pipeline_materialization(
    const std::span<const std::uint8_t> barriers,
    const std::span<const std::uint32_t> declared_steps,
    const std::span<const BackendPublish> publications,
    const std::uint32_t declared_step_count,
    const std::uint32_t generation_stride, const bool profile_steps,
    PreparedKernelTemplateRegistry &registry,
    PipelineBudgetTransaction &budget_transaction,
    PipelineMaterializationDraft &draft) noexcept {
  prepared::PipelineState &pipeline = *draft.pipeline;
  try {
    if (!expand_pipeline(draft.batch_templates, barriers, publications,
                         declared_steps, declared_step_count, profile_steps,
                         pipeline.ops->nested_aggregate_command_count,
                         draft.expanded)) {
      return reject_pipeline(draft.expanded.failure, draft.expanded.reason);
    }
  } catch (const std::bad_alloc &) {
    return reject_pipeline(draft.expanded.failure, "compute_pipeline_capacity");
  }
  const MapRecurrence common_recurrence =
      BuildMapRecurrence(draft.expanded.commands, draft.expanded.barriers);
  if (common_recurrence.invalid()) {
    return reject_pipeline(draft.expanded.failure, common_recurrence.reason);
  }
  if (common_recurrence.ready()) {
    if (draft.expanded.commands.empty() ||
        draft.expanded.commands.front().template_index >= pipeline.size ||
        pipeline.states[draft.expanded.commands.front().template_index] ==
            nullptr) {
      return reject_pipeline(draft.expanded.failure,
                             "accel_kernel_run_invalid");
    }
    const auto semantic_owner = std::static_pointer_cast<const void>(
        pipeline.states[draft.expanded.commands.front().template_index]);
    ServiceFreeDirectProjection projected = ProjectServiceFreeDirectProof(
        common_recurrence, draft.expanded.commands, semantic_owner,
        draft.reservation.fingerprint_hi, draft.reservation.fingerprint_lo);
    if (!projected.check.ok || projected.proof == nullptr ||
        !service_free_direct_proof_valid(*projected.proof)) {
      return reject_pipeline(draft.expanded.failure,
                             projected.check.reason == nullptr
                                 ? "accel_kernel_pipeline_invalid"
                                 : projected.check.reason);
    }
    pipeline.service_free_direct = std::move(projected.proof);
  }
  draft.failure.stage(PreparedPipelineFailureStage::CommonAccounting);
  if (const char *const reason = account_pipeline(
          declared_steps, declared_step_count, generation_stride, draft);
      reason != nullptr) {
    return reject_pipeline(draft.failure, reason);
  }
  PreparedPipelineFailure backend_failure{};
  const rund::AccelCheck built = pipeline.ops->prepare_pipeline(
      draft.batch_templates, draft.expanded.commands, draft.expanded.barriers,
      draft.expanded.transducers, draft.expanded.aggregates, publications,
      registry, pipeline.status, profile_steps, pipeline.backend,
      draft.backend_memory, &pipeline.memory, draft.preparation,
      backend_failure);
  if (!built.ok) {
    if (backend_failure.stage == PreparedPipelineFailureStage::Unknown) {
      draft.failure.stage(PreparedPipelineFailureStage::Unknown);
      backend_failure = draft.failure.failure(built.reason);
    }
    return PreparedKernelPipeline{.failure = backend_failure};
  }
  if (pipeline.service_free_direct != nullptr) {
    const std::shared_ptr<prepared::RunState> semantic = pipeline.states[0];
    if (semantic == nullptr) {
      return reject_pipeline(draft.failure, "accel_kernel_pipeline_invalid");
    }
    try {
      auto compact =
          std::make_unique<std::shared_ptr<prepared::RunState>[]>(1u);
      compact[0] = semantic;
      pipeline.states = std::move(compact);
      pipeline.state_count = 1u;
    } catch (const std::bad_alloc &) {
      return reject_pipeline(draft.failure, "compute_pipeline_capacity");
    }
  }
  draft.failure.stage(PreparedPipelineFailureStage::CommonAccounting);
  if (!ValidPreparedPipelineStatusLayout(
          pipeline.status, declared_steps, declared_step_count,
          draft.expanded.command_count, generation_stride)) {
    return reject_pipeline(draft.failure, "accel_kernel_run_invalid");
  }
  std::uint64_t state_owner_bytes = 0u;
  std::uint64_t common_host_bytes = 0u;
  if (!mul(pipeline.state_count, sizeof(std::shared_ptr<prepared::RunState>),
           state_owner_bytes) ||
      !add(sizeof(prepared::PipelineState), state_owner_bytes,
           common_host_bytes) ||
      (pipeline.service_free_direct != nullptr &&
       !add(common_host_bytes, draft.reservation.service_free_direct_host_bytes,
            common_host_bytes))) {
    return reject_pipeline(draft.failure, "compute_pipeline_capacity");
  }
  accumulate_memory(draft.backend_memory.host,
                    PreparedMemory{.current = common_host_bytes,
                                   .peak = common_host_bytes,
                                   .cumulative = common_host_bytes,
                                   .budget = common_host_bytes});
  pipeline.memory.add(draft.backend_memory);
  budget_transaction.commit();
  return PreparedKernelPipeline{
      .owner = std::static_pointer_cast<void>(draft.pipeline),
      .preparation = draft.preparation,
      .ok = true,
  };
}

} // namespace rund::node::accel::detail
