#include "../prepared.hpp"

#include "model.hpp"

#include "../run/shape.hpp"

#include <rund/counter.hpp>

#include <memory>
#include <new>

namespace rund::node::accel::detail {

PreparedKernelRun PrepareKernelRun(const rund::AccelContext &context,
                                   const rund::AccelKernel &kernel,
                                   const rund::AccelRun &run,
                                   const KernelPreparationMode mode,
                                   const KernelViewLayout *const views,
                                   const RunBinds *const view_binds,
                                   const KernelScratchLayout *const scratch) {
  std::shared_ptr<prepared::RunState> state{};
  try {
    state = std::make_shared<prepared::RunState>();
  } catch (const std::bad_alloc &) {
    return {};
  }
  state->mode = mode;
  state->tile_count = run.tile_count;
  state->execution = AdmitKernelForExecution(context, kernel);
  if (!state->execution.admission.check.ok ||
      !RunRequestShapeOk(context, state->execution, run)) {
    return {};
  }
  state->binds = BuildRunBinds(state->execution, run);
  if (!state->binds.ok) {
    return PreparedKernelRun{.reason = state->binds.reason};
  }
  state->resets = BuildResetBinds(state->execution, run, state->binds.binds);
  if (!state->resets.ok) {
    return PreparedKernelRun{.reason = state->resets.reason};
  }
  state->dispatch = BuildRunDispatch(state->execution, run);
  if (!state->dispatch.ok) {
    return PreparedKernelRun{.reason = state->dispatch.reason};
  }
  state->order = BuildScheduledStepOrder(
      state->execution.steps, state->execution.graph_roles,
      state->execution.graph_alias_representatives,
      state->execution.required_barriers);
  if (!state->order.ok ||
      state->order.size() != state->execution.steps.size()) {
    return {};
  }
  state->roundtrip =
      CountProducerConsumerRoundtripBytes(state->execution, state->order, run);
  if (!state->roundtrip.ok) {
    return PreparedKernelRun{.reason = state->roundtrip.reason};
  }
  state->bound =
      BuildBoundRun(context, state->execution, state->binds.binds,
                    state->resets.binds, state->dispatch.planned_steps,
                    state->order, state->dispatch.original_dispatch_count,
                    state->dispatch.final_dispatch_count);
  state->bound.run.resets = &state->resets.binds;
  state->bound.run.views = views;
  state->bound.run.view_binds = view_binds;
  state->bound.run.scratch = scratch;
  if (IsPipelinePrivatePreparation(mode) &&
      ((scratch == nullptr) != (view_binds == nullptr) ||
       (scratch != nullptr && !ValidKernelScratch(*scratch, *view_binds)))) {
    return PreparedKernelRun{.reason = "accel_kernel_scratch_invalid"};
  }
  if (!state->bound.ok || state->bound.run.ops == nullptr ||
      state->bound.run.ops->submit_prepared == nullptr) {
    return PreparedKernelRun{.reason = "accel_kernel_prepared_incomplete"};
  }
  // Pipeline-private preparation stops at the immutable host route. The
  // enclosing Pipeline must preflight every route/template contribution as
  // one checked reservation before any backend or native owner is created.
  if (IsPipelinePrivatePreparation(mode)) {
    if (state->bound.run.ops->plan_pipeline_private == nullptr ||
        state->bound.run.ops->same_pipeline_template == nullptr ||
        state->bound.run.ops->prepare_pipeline_private == nullptr) {
      return PreparedKernelRun{.reason = "accel_kernel_prepared_incomplete"};
    }
    return PreparedKernelRun{.owner = std::static_pointer_cast<void>(state),
                             .ok = true,
                             .reason = "ok"};
  }
  PreparedMemory memory{};
  const auto prepare = state->bound.run.ops->prepare;
  if (prepare == nullptr) {
    return PreparedKernelRun{.reason = "accel_kernel_prepared_incomplete"};
  }
  std::uint32_t failed_node = NoNode;
  state->bound.run.failed_node = &failed_node;
  const rund::AccelCheck prepared =
      prepare(state->bound.run, state->backend, memory);
  state->bound.run.failed_node = nullptr;
  if (!prepared.ok) {
    return PreparedKernelRun{.reason = prepared.reason,
                             .failed_node = failed_node};
  }
  if (state->bound.run.ops->traffic != nullptr) {
    state->roundtrip.internal_bytes = ::rund::detail::counter::SaturatingAdd(
        state->roundtrip.internal_bytes,
        state->bound.run.ops->traffic(state->backend));
  }
  state->memory.add(memory);
  return PreparedKernelRun{.owner = std::static_pointer_cast<void>(state),
                           .ok = true,
                           .reason = "ok"};
}

PreparedMemory
ReadPreparedKernelMemory(const PreparedKernelRun &prepared) noexcept {
  const auto *const state =
      static_cast<const prepared::RunState *>(prepared.owner.get());
  return prepared.ok && state != nullptr ? state->memory.read()
                                         : PreparedMemory{};
}

} // namespace rund::node::accel::detail
