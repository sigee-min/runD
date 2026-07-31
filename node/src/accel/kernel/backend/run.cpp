#include "run.hpp"

#include "../../backend/token.hpp"
#include "../bindings.hpp"
#include "../bindings/local.hpp"
#include "../step/map/local.hpp"

#include <accel/api.hpp>

#include <utility>

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] bool BindPrimitive(const KernelExecutionStep &step,
                                 const RunBinds &run_binds,
                                 BoundBindings &out) {
  switch (step.kind()) {
  case rund::kernel::NodeKind::Map: {
    StepBinds bindings{};
    if (!BuildStepBinds(step, run_binds, bindings)) {
      return false;
    }
    out = std::move(bindings);
    return true;
  }
  case rund::kernel::NodeKind::Scan: {
    ScanBinds bindings{};
    if (!BuildScanBinds(step, run_binds, bindings)) {
      return false;
    }
    out = std::move(bindings);
    return true;
  }
  case rund::kernel::NodeKind::Compact: {
    CompactBinds bindings{};
    if (!BuildCompactBinds(step, run_binds, bindings)) {
      return false;
    }
    out = bindings;
    return true;
  }
  case rund::kernel::NodeKind::SegmentedScan: {
    SegmentedScanBinds bindings{};
    if (!BuildSegmentedScanBinds(step, run_binds, bindings)) {
      return false;
    }
    out = std::move(bindings);
    return true;
  }
  case rund::kernel::NodeKind::SegmentedReduce: {
    SegmentedReduceBinds bindings{};
    if (!BuildSegmentedReduceBinds(step, run_binds, bindings)) {
      return false;
    }
    out = std::move(bindings);
    return true;
  }
  case rund::kernel::NodeKind::Sort: {
    SortBinds bindings{};
    if (!BuildSortBinds(step, run_binds, bindings)) {
      return false;
    }
    out = std::move(bindings);
    return true;
  }
  case rund::kernel::NodeKind::Gather: {
    GatherBinds bindings{};
    if (!BuildGatherBinds(step, run_binds, bindings)) {
      return false;
    }
    out = std::move(bindings);
    return true;
  }
  case rund::kernel::NodeKind::Histogram: {
    HistogramBinds bindings{};
    if (!BuildHistogramBinds(step, run_binds, bindings)) {
      return false;
    }
    out = std::move(bindings);
    return true;
  }
  case rund::kernel::NodeKind::Partition: {
    PartitionBinds bindings{};
    if (!BuildPartitionBinds(step, run_binds, bindings)) {
      return false;
    }
    out = std::move(bindings);
    return true;
  }
  case rund::kernel::NodeKind::Reduce: {
    ReduceBinds bindings{};
    if (!BuildReduceBinds(step, run_binds, bindings)) {
      return false;
    }
    out = std::move(bindings);
    return true;
  }
  case rund::kernel::NodeKind::Scatter: {
    ScatterBinds bindings{};
    if (!BuildScatterBinds(step, run_binds, bindings)) {
      return false;
    }
    out = std::move(bindings);
    return true;
  }
  case rund::kernel::NodeKind::ScatterReduce: {
    ScatterReduceBinds bindings{};
    if (!BuildScatterReduceBinds(step, run_binds, bindings))
      return false;
    out = std::move(bindings);
    return true;
  }
  case rund::kernel::NodeKind::Stencil: {
    StencilBinds bindings{};
    if (!BuildStencilBinds(step, run_binds, bindings)) {
      return false;
    }
    out = std::move(bindings);
    return true;
  }
  case rund::kernel::NodeKind::Transform: {
    TransformBinds bindings{};
    if (!BuildTransformBinds(step, run_binds, bindings)) {
      return false;
    }
    out = std::move(bindings);
    return true;
  }
  case rund::kernel::NodeKind::Matrix: {
    MatrixBinds bindings{};
    if (!BuildMatrixBinds(step, run_binds, bindings)) {
      return false;
    }
    out = std::move(bindings);
    return true;
  }
  case rund::kernel::NodeKind::Factor: {
    FactorBinds bindings{};
    if (!BuildFactorBinds(step, run_binds, bindings)) {
      return false;
    }
    out = std::move(bindings);
    return true;
  }
  case rund::kernel::NodeKind::Solve: {
    SolveBinds bindings{};
    if (!BuildSolveBinds(step, run_binds, bindings)) {
      return false;
    }
    out = std::move(bindings);
    return true;
  }
  case rund::kernel::NodeKind::Spectrum: {
    SpectrumBinds bindings{};
    if (!BuildSpectrumBinds(step, run_binds, bindings)) {
      return false;
    }
    out = std::move(bindings);
    return true;
  }
  default:
    return false;
  }
}

[[nodiscard]] bool BindMap(const rund::AccelContext &context,
                           BoundStep &bound) {
  if (bound.step == nullptr || bound.planned == nullptr ||
      bound.step->kind() != rund::kernel::NodeKind::Map) {
    return true;
  }
  const StepBinds *const bindings = BindingsFor<StepBinds>(bound);
  if (bindings == nullptr) {
    return false;
  }
  const rund::kernel::BindingSet map_binding =
      BindMapStep(*bound.step, *bound.planned, *bindings);
  const rund::kernel::BindingValidation validation =
      ValidateMapStepBindings(*bound.step, *bound.planned, map_binding);
  if (!validation.ok) {
    return false;
  }
  const bool resident_identity = map_binding.has_resident_output() &&
                                 map_binding.sequence_tiles == nullptr &&
                                 map_binding.sequence_tile_count == 0u;
  bound.map_windows =
      ResidentDispatchWindows(bound.planned->plan, resident_identity,
                              context.api != rund::AccelApi::Fake &&
                                  context.api != rund::AccelApi::Cpu);
  return bound.map_windows.ok;
}

[[nodiscard]] bool BindControl(const KernelExecutionStep &step,
                               const RunBinds &run_binds,
                               BoundControl &out) noexcept {
  out = BoundControl{.control = step.control};
  if (!step.control.valid(step.graph_binding_indices.size())) {
    return false;
  }
  const BindingSource source = BindingSourceFor(run_binds);
  const auto bind = [&](const std::uint32_t local,
                        const rund::kernel::ResidentBufferRef *&ref,
                        const std::shared_ptr<void> *&handle) {
    return local < step.graph_binding_indices.size() &&
           ReadBinding(source, step.graph_binding_indices[local], ref,
                       handle) &&
           BindingReady(handle);
  };
  return (!step.control.has_count() ||
          bind(step.control.count_binding, out.count, out.count_handle)) &&
         (!step.control.has_predicate() ||
          bind(step.control.predicate_binding, out.predicate,
               out.predicate_handle));
}

} // namespace

bool RebindBoundStep(const BoundStep &source, const RunBinds &binds,
                     BoundStep &out) {
  if (source.step == nullptr || source.planned == nullptr) {
    return false;
  }
  out = BoundStep{.index = source.index,
                  .step = source.step,
                  .planned = source.planned,
                  .source_binds = &binds,
                  .barrier_before = source.barrier_before};
  if (!BindPrimitive(*source.step, binds, out.bindings) ||
      !BindControl(*source.step, binds, out.control)) {
    return false;
  }
  // Dense View lowering changes bindings, not the frozen dispatch frontier.
  if (source.step->kind() == rund::kernel::NodeKind::Map) {
    out.map_windows = source.map_windows;
  }
  return true;
}

BoundRun::BoundRun(BoundRun &&other) noexcept
    : run(other.run), storage(std::move(other.storage)), ok(other.ok),
      reason(other.reason) {
  if (run.execution != nullptr) {
    run.steps = storage.data();
  }
  other.run = {};
  other.ok = false;
}

BoundRun &BoundRun::operator=(BoundRun &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  run = other.run;
  storage = std::move(other.storage);
  ok = other.ok;
  reason = other.reason;
  if (run.execution != nullptr) {
    run.steps = storage.data();
  }
  other.run = {};
  other.ok = false;
  return *this;
}

void BoundRun::bind(const KernelExecution &execution,
                    const std::uint64_t original_dispatch_count,
                    const std::uint64_t final_dispatch_count) noexcept {
  const std::shared_ptr<PickToken> &token = execution.context_admission.pick;
  run = BackendRun{
      .pick = token == nullptr ? nullptr : &token->raw,
      .ops = token == nullptr ? nullptr : token->ops,
      .execution = &execution,
      .steps = storage.data(),
      .step_count = storage.size(),
      .original_dispatch_count = original_dispatch_count,
      .final_dispatch_count = final_dispatch_count,
  };
}

BoundRun BuildBoundRun(const rund::AccelContext &context,
                       const KernelExecution &execution, const RunBinds &binds,
                       const BoundResets &resets,
                       const PlannedStepStorage &planned,
                       const ScheduledStepOrder &order,
                       const std::uint64_t original_dispatch_count,
                       const std::uint64_t final_dispatch_count) {
  BoundRun result{};
  if (!order.ok || order.size() != execution.steps.size() || !planned.valid() ||
      planned.size() != execution.steps.size() ||
      !execution.context_admission.check.ok ||
      execution.context_admission.pick == nullptr ||
      execution.context_admission.pick->ops == nullptr ||
      !execution.context_admission.pick->raw.check.ok ||
      execution.context_admission.pick->raw.api != context.api ||
      execution.context_admission.pick->ops->api != context.api) {
    return result;
  }
  result.storage.reserve(order.size());
  std::size_t reset_cursor = 0u;
  for (std::size_t position = 0u; position < order.size(); ++position) {
    const std::size_t index = order.at(position);
    if (index >= execution.steps.size()) {
      return result;
    }
    const KernelExecutionStep &step = execution.steps[index];
    const PlannedStep *const step_plan = planned.get(index);
    if (step_plan == nullptr) {
      return result;
    }
    if (reset_cursor < resets.size() &&
        resets[reset_cursor].step.index < index) {
      return result;
    }
    const std::size_t reset_begin = reset_cursor;
    while (reset_cursor < resets.size() &&
           resets[reset_cursor].step.index == index) {
      ++reset_cursor;
    }
    BoundStep bound{.index = index,
                    .step = &step,
                    .planned = step_plan,
                    .source_binds = &binds,
                    .resets =
                        ResetSpan{
                            .begin = reset_begin,
                            .count = reset_cursor - reset_begin,
                        },
                    .barrier_before = order.barrier_before(position)};
    if (!BindPrimitive(step, binds, bound.bindings) ||
        !BindControl(step, binds, bound.control)) {
      return result;
    }
    result.storage.push_back(std::move(bound), order.size());
    BoundStep *const stored = result.storage.get(position);
    if (stored == nullptr || !BindMap(context, *stored)) {
      return result;
    }
  }
  if (!result.storage.valid() || reset_cursor != resets.size()) {
    return result;
  }
  result.bind(execution, original_dispatch_count, final_dispatch_count);
  if (result.run.pick == nullptr || result.run.ops == nullptr) {
    return result;
  }
  result.ok = true;
  result.reason = "ok";
  return result;
}

rund::kernel::BindingSet MapBindingFor(const BoundStep &step) noexcept {
  if (step.step == nullptr || step.planned == nullptr ||
      step.step->kind() != rund::kernel::NodeKind::Map) {
    return {};
  }
  const StepBinds *const bindings = BindingsFor<StepBinds>(step);
  return bindings == nullptr
             ? rund::kernel::BindingSet{}
             : BindMapStep(*step.step, *step.planned, *bindings);
}

} // namespace rund::node::accel::detail
