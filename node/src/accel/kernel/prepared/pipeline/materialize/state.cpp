#include "../../run.hpp"
#include "internal.hpp"

#include <new>

namespace rund::node::accel::detail {

namespace {

class BackendPreparationCursor final {
public:
  BackendPreparationCursor(BackendRun &run,
                           PreparedKernelTemplateRegistry &templates,
                           std::uint32_t &failed_node,
                           const BackendTemplateRouteDemand demand) noexcept
      : run_{run} {
    run_.templates = &templates;
    run_.template_route_demand = demand;
    run_.failed_node = &failed_node;
  }

  ~BackendPreparationCursor() {
    run_.templates = nullptr;
    run_.template_route_demand = {};
    run_.failed_node = nullptr;
  }

  BackendPreparationCursor(const BackendPreparationCursor &) = delete;
  BackendPreparationCursor &
  operator=(const BackendPreparationCursor &) = delete;

private:
  BackendRun &run_;
};

} // namespace

bool BackendPreparationCursorLifecycleForContract(
    BackendRun &run, const BackendTemplateRouteDemand demand) noexcept {
  if (run.templates != nullptr || run.failed_node != nullptr ||
      !run.template_route_demand.empty() || !demand.valid()) {
    return false;
  }
  PreparedKernelTemplateRegistry templates{};
  std::uint32_t failed_node = NoNode;
  const bool escaped = [&]() noexcept {
    const BackendPreparationCursor cursor{run, templates, failed_node, demand};
    if (run.templates != &templates || run.failed_node != &failed_node ||
        run.template_route_demand.owner_count != demand.owner_count ||
        run.template_route_demand.route_copies != demand.route_copies ||
        run.template_route_demand.capacity != demand.capacity) {
      return true;
    }
    // Exercise destruction across an early-return edge, matching every
    // fail-closed return from private backend materialization.
    return false;
  }();
  return !escaped && run.templates == nullptr && run.failed_node == nullptr &&
         run.template_route_demand.empty();
}

const char *materialize_pipeline_routes(
    const rund::AccelContext &context,
    const std::span<const PreparedKernelRun *const> runs,
    const std::span<const BackendRecurrence> recurrences,
    PreparedKernelTemplateRegistry &registry,
    PipelineMaterializationDraft &draft) noexcept {
  try {
    draft.pipeline = std::make_shared<prepared::PipelineState>();
    draft.pipeline->states =
        std::make_unique<std::shared_ptr<prepared::RunState>[]>(runs.size());
  } catch (const std::bad_alloc &) {
    return "compute_pipeline_capacity";
  }
  prepared::PipelineState &pipeline = *draft.pipeline;
  pipeline.context = context;
  pipeline.templates = registry.owner;
  pipeline.state_count = runs.size();
  pipeline.size = runs.size();
  try {
    draft.batch_templates.resize(runs.size());
    for (std::size_t index = 0u; index < runs.size(); ++index) {
      draft.failure.compact_template_route(static_cast<std::uint32_t>(index),
                                           recurrences[index]);
      const PreparedKernelRun *const item = runs[index];
      auto *const state =
          item == nullptr
              ? nullptr
              : static_cast<prepared::RunState *>(item->owner.get());
      const BackendOps *const candidate =
          state == nullptr ? nullptr : state->bound.run.ops;
      if (item == nullptr || !item->ok || state == nullptr ||
          !IsPipelinePrivatePreparation(state->mode) || candidate == nullptr ||
          candidate->prepare_pipeline == nullptr ||
          candidate->submit_prepared_pipeline == nullptr ||
          !prepared::MatchesContext(context, *state) ||
          (pipeline.ops != nullptr && pipeline.ops != candidate)) {
        return "accel_kernel_run_invalid";
      }
      pipeline.ops = candidate;
      pipeline.states[index] =
          std::static_pointer_cast<prepared::RunState>(item->owner);
      const BackendTemplateRouteDemand demand =
          draft.template_route_demands[index];
      if (!demand.valid() || state->bound.run.templates != nullptr ||
          state->bound.run.failed_node != nullptr ||
          !state->bound.run.template_route_demand.empty()) {
        return "accel_kernel_template_invalid";
      }
      if (state->backend == nullptr) {
        PreparedMemory memory{};
        std::uint32_t failed_node = NoNode;
        const BackendPreparationCursor cursor{state->bound.run, registry,
                                              failed_node, demand};
        const rund::AccelCheck materialized =
            candidate->prepare_pipeline_private(state->bound.run,
                                                state->backend, memory);
        if (!materialized.ok || state->backend == nullptr) {
          draft.failure_override = draft.failure.failure(materialized.reason);
          draft.failure_override.node = failed_node;
          draft.has_failure_override = true;
          return materialized.reason == nullptr
                     ? "accel_kernel_pipeline_invalid"
                     : materialized.reason;
        }
        if (candidate->traffic != nullptr) {
          state->roundtrip.internal_bytes =
              ::rund::detail::counter::SaturatingAdd(
                  state->roundtrip.internal_bytes,
                  candidate->traffic(state->backend));
        }
        state->memory.add(memory);
      }
      draft.batch_templates[index] = BackendBatchEntry{
          .run = &state->bound.run,
          .prepared = &state->backend,
          .recurrence = recurrences[index],
          .template_index = static_cast<std::uint32_t>(index)};
    }
  } catch (const std::bad_alloc &) {
    return "compute_pipeline_capacity";
  }
  return nullptr;
}

} // namespace rund::node::accel::detail
