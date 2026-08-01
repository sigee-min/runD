#include "../recurrence.hpp"

#include "match.hpp"

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/gather/plan.hpp>
#include <kernel/program/compute/reduce/plan.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace rund::node::accel::detail {
namespace {

using rund::kernel::BindingSet;
using rund::kernel::ComputeApi;
using rund::kernel::ComputeCountSource;
using rund::kernel::ComputeDomain;
using rund::kernel::ComputeScalar;
using rund::kernel::ResidentBindingRange;
using rund::kernel::ResidentBufferRef;

struct View final {
  const ResidentBufferRef *ref{};
  const std::shared_ptr<void> *handle{};

  [[nodiscard]] bool valid() const noexcept {
    return ref != nullptr && handle != nullptr && *handle != nullptr;
  }
};

struct SeedProjection final {
  BackendRead queue{};
  BackendRead domain{};
  BackendRead count{};
  View tile_low{};
  View tile_status{};
  View tile_state{};
  View tile_count{};
  std::uint32_t invalid_index_source_node{NoNode};
  std::uint32_t reduce_overflow_source_node{NoNode};
};

// Cold-only identity used while proving that every authored template belongs
// to one admitted program. It is deliberately not retained in the aggregate:
// native execution consumes the normalized proof, never an execution owner.
struct ProgramFingerprint final {
  rund::AccelApi api{rund::AccelApi::Auto};
  std::uint64_t graph_id_hi{};
  std::uint64_t graph_id_lo{};
  ComputeScalar scalar{ComputeScalar::Lane32};
  ComputeDomain domain{ComputeDomain::Fixed};
  rund::kernel::ComputeFixedFormat fixed_format{};
};

[[nodiscard]] NestedAggregate Ineligible(const char *const reason) noexcept {
  return NestedAggregate{.reason = reason};
}

[[nodiscard]] NestedAggregate Invalid(const char *const reason) noexcept {
  return NestedAggregate{
      .state = NestedAggregateState::Invalid,
      .reason = reason,
  };
}

[[nodiscard]] View At(const ResidentBindingRange &range,
                      const std::uint64_t index) noexcept {
  return View{.ref = range.ref(index), .handle = range.handle(index)};
}

[[nodiscard]] View At(const ResidentBufferRef *const ref,
                      const std::shared_ptr<void> *const handle) noexcept {
  return View{.ref = ref, .handle = handle};
}

[[nodiscard]] bool SameStorage(const View left, const View right) noexcept {
  return left.valid() && right.valid() && left.ref->id == right.ref->id &&
         left.ref->bytes == right.ref->bytes &&
         left.ref->offset_bytes == right.ref->offset_bytes &&
         left.ref->element_bytes == right.ref->element_bytes &&
         left.ref->stride_bytes == right.ref->stride_bytes &&
         left.ref->count == right.ref->count && *left.handle == *right.handle;
}

[[nodiscard]] bool ReadView(const View view) noexcept {
  return view.valid() && view.ref->usage == rund::kernel::kResidentUsageRead;
}

[[nodiscard]] bool WriteView(const View view) noexcept {
  return view.valid() && view.ref->usage == rund::kernel::kResidentUsageWrite;
}

[[nodiscard]] bool U32View(const View view,
                           const std::uint64_t count) noexcept {
  return view.valid() && view.ref->element_bytes == sizeof(std::uint32_t) &&
         view.ref->stride_bytes >= sizeof(std::uint32_t) &&
         view.ref->count == count &&
         (view.ref->offset_bytes % sizeof(std::uint32_t)) == 0u &&
         (view.ref->stride_bytes % sizeof(std::uint32_t)) == 0u;
}

[[nodiscard]] bool DenseU32Workspace(const View view,
                                     const std::uint64_t count) noexcept {
  if (!WriteView(view) || count == 0u || view.ref->id == 0u ||
      view.ref->element_bytes != sizeof(std::uint32_t) ||
      view.ref->stride_bytes != sizeof(std::uint32_t) ||
      view.ref->count < count ||
      (view.ref->offset_bytes % sizeof(std::uint32_t)) != 0u ||
      view.ref->offset_bytes > view.ref->bytes) {
    return false;
  }
  return view.ref->count <=
         (view.ref->bytes - view.ref->offset_bytes) / sizeof(std::uint32_t);
}

[[nodiscard]] bool InternalOutput(const BackendRun &run,
                                  const BoundStep &step) noexcept {
  if (run.execution == nullptr || step.step == nullptr ||
      !step.step->graph_binding_indices_ok ||
      step.step->graph_binding_indices.size() == 0u) {
    return false;
  }
  const std::uint64_t binding =
      step.step
          ->graph_binding_indices[step.step->graph_binding_indices.size() - 1u];
  return binding < run.execution->graph_visibilities.size() &&
         run.execution->graph_visibilities[static_cast<std::size_t>(binding)] ==
             rund::GraphBufferVisibility::Internal;
}

[[nodiscard]] bool DisjointStorage(const View left, const View right) noexcept {
  if (!left.valid() || !right.valid()) {
    return false;
  }
  const bool same_id = left.ref->id == right.ref->id;
  const bool same_owner = *left.handle == *right.handle;
  if (same_id != same_owner) {
    return false;
  }
  if (!same_id) {
    return true;
  }
  const auto end = [](const View view, std::uint64_t &out) {
    std::uint64_t tail = 0u;
    return view.ref->count != 0u &&
           rund::kernel::checked::mul(view.ref->count - 1u,
                                      view.ref->stride_bytes, tail) &&
           rund::kernel::checked::add(view.ref->offset_bytes, tail, out) &&
           rund::kernel::checked::add(out, view.ref->element_bytes, out);
  };
  std::uint64_t left_end = 0u;
  std::uint64_t right_end = 0u;
  return end(left, left_end) && end(right, right_end) &&
         (left_end <= right.ref->offset_bytes ||
          right_end <= left.ref->offset_bytes);
}

[[nodiscard]] NestedAggregateWorkspace Workspace(const View view) {
  return NestedAggregateWorkspace{.ref = *view.ref, .handle = *view.handle};
}

[[nodiscard]] BackendRead Read(const View view) {
  return BackendRead{.source = *view.ref, .handle = *view.handle};
}

[[nodiscard]] bool SameRead(const BackendRead &left,
                            const BackendRead &right) noexcept {
  return SameStorage(At(&left.source, &left.handle),
                     At(&right.source, &right.handle));
}

[[nodiscard]] bool SameWindowIdentity(const BackendWindow &left,
                                      const BackendWindow &right) noexcept {
  if (left.maximum != right.maximum || left.tile != right.tile ||
      left.expected != right.expected || left.state != right.state ||
      left.outer_bound != right.outer_bound ||
      left.inner_bound != right.inner_bound ||
      left.has_terminal != right.has_terminal ||
      !SameRead(left.count, right.count)) {
    return false;
  }
  for (std::size_t bank = 0u; bank < left.terminal.size(); ++bank) {
    if (left.has_terminal &&
        !SameRead(left.terminal[bank], right.terminal[bank])) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool ReadyRun(const BackendRun *const run,
                            const std::size_t step_count) noexcept {
  return run != nullptr && run->execution != nullptr && run->steps != nullptr &&
         run->step_count == step_count &&
         (run->resets == nullptr || run->resets->empty());
}

[[nodiscard]] bool ReadyStep(const BoundStep &step,
                             const rund::kernel::NodeKind kind,
                             const bool first) noexcept {
  return BoundStepMatches(step, kind) && step.step != nullptr &&
         step.planned != nullptr && step.source_binds != nullptr &&
         !step.control.active() && step.resets.empty() &&
         step.barrier_before == !first;
}

[[nodiscard]] ProgramFingerprint
ProgramIdentity(const BackendRun &run) noexcept {
  if (run.execution == nullptr || run.execution->steps.empty()) {
    return {};
  }
  return ProgramFingerprint{
      .api = run.execution->admission.api,
      .graph_id_hi = run.execution->admission.graph_id_hi,
      .graph_id_lo = run.execution->admission.graph_id_lo,
      .scalar = run.execution->admission.scalar,
      .domain = run.execution->admission.domain,
      .fixed_format = run.execution->steps.front().artifact.key.fixed_format,
  };
}

[[nodiscard]] bool SameProgram(const BackendRun &run,
                               const ProgramFingerprint &identity) noexcept {
  if (run.execution == nullptr ||
      run.execution->admission.api != identity.api ||
      run.execution->admission.graph_id_hi != identity.graph_id_hi ||
      run.execution->admission.graph_id_lo != identity.graph_id_lo ||
      run.execution->admission.scalar != identity.scalar ||
      run.execution->admission.domain != identity.domain ||
      run.execution->steps.empty()) {
    return false;
  }
  return run.execution->steps.front().artifact.key.fixed_format ==
         identity.fixed_format;
}

[[nodiscard]] bool U32Program(const ProgramFingerprint &identity) noexcept {
  return identity.api != rund::AccelApi::Cpu &&
         (identity.api == rund::AccelApi::Metal ||
          identity.api == rund::AccelApi::Vulkan) &&
         identity.scalar == ComputeScalar::Lane32 &&
         identity.domain == ComputeDomain::U32 &&
         (identity.graph_id_hi != 0u || identity.graph_id_lo != 0u);
}

[[nodiscard]] bool MapShape(const BoundStep &step,
                            const MapSemanticKind semantic,
                            const std::uint64_t inputs,
                            const std::uint64_t outputs,
                            BindingSet &bindings) noexcept {
  if (!BoundStepMatches(step, rund::kernel::NodeKind::Map) ||
      step.step == nullptr || step.planned == nullptr ||
      step.source_binds == nullptr || step.control.active() ||
      !step.resets.empty() || step.step->map_semantic.kind != semantic ||
      !step.step->artifact.ok || !step.step->graph_binding_indices_ok ||
      !step.planned->plan.ok || step.planned->artifact == nullptr ||
      step.planned->artifact != &step.step->artifact ||
      step.step->artifact.key.scalar != ComputeScalar::Lane32 ||
      step.step->artifact.key.domain != ComputeDomain::U32) {
    return false;
  }
  bindings = MapBindingFor(step);
  return bindings.ok && bindings.resident_inputs.count == inputs &&
         bindings.resident_outputs.count == outputs &&
         bindings.resident_inputs.has_refs() &&
         bindings.resident_inputs.has_handles() &&
         bindings.resident_outputs.has_refs() &&
         bindings.resident_outputs.has_handles();
}

[[nodiscard]] bool CollectiveStep(const BoundStep &step,
                                  const rund::kernel::NodeKind kind) noexcept {
  return ReadyStep(step, kind, false) && step.step->source.valid();
}

[[nodiscard]] bool GatherShape(const BoundStep &step, const std::uint32_t tile,
                               View &values, View &indices, View &count,
                               View &output) noexcept {
  const operation::Gather *const operation =
      OperationFor<operation::Gather>(step);
  const GatherBinds *const bindings = BindingsFor<GatherBinds>(step);
  if (!CollectiveStep(step, rund::kernel::NodeKind::Gather) ||
      operation == nullptr || bindings == nullptr ||
      operation->desc.element != rund::kernel::GatherElement::U32 ||
      operation->desc.element_count != tile ||
      operation->desc.count_source != ComputeCountSource::BufferU32 ||
      !rund::kernel::GatherPlanMatchesDesc(operation->desc, operation->plan)) {
    return false;
  }
  values = At(bindings->values, bindings->values_handle);
  indices = At(bindings->indices, bindings->indices_handle);
  count = At(bindings->logical_count, bindings->logical_count_handle);
  output = At(bindings->output, bindings->output_handle);
  return ReadView(values) && ReadView(indices) && ReadView(count) &&
         WriteView(output) && U32View(values, operation->desc.source_count) &&
         U32View(indices, tile) && U32View(count, 1u) && U32View(output, tile);
}

[[nodiscard]] bool ReduceShape(const BoundStep &step, const std::uint32_t tile,
                               View &input, View &count,
                               View &output) noexcept {
  const operation::Reduce *const operation =
      OperationFor<operation::Reduce>(step);
  const ReduceBinds *const bindings = BindingsFor<ReduceBinds>(step);
  if (!CollectiveStep(step, rund::kernel::NodeKind::Reduce) ||
      operation == nullptr || bindings == nullptr ||
      operation->desc.op != rund::kernel::ReduceOp::Sum ||
      operation->desc.element != rund::kernel::ReduceElement::U32 ||
      operation->desc.element_count != tile ||
      operation->desc.count_source != ComputeCountSource::BufferU32 ||
      !rund::kernel::ReducePlanMatchesDesc(operation->desc, operation->plan)) {
    return false;
  }
  input = At(bindings->input, bindings->input_handle);
  count = At(bindings->logical_count, bindings->logical_count_handle);
  output = At(bindings->output, bindings->output_handle);
  return ReadView(input) && ReadView(count) && WriteView(output) &&
         U32View(input, tile) && U32View(count, 1u) && U32View(output, 1u);
}

[[nodiscard]] bool SeedShape(const BackendRun &run, const BackendWindow &window,
                             SeedProjection &projection, const char *&reason) {
  if (!ReadyRun(&run, 6u)) {
    reason = "compute_pipeline_nested_aggregate_seed_run_ineligible";
    return false;
  }
  constexpr std::array<bool, 6u> barriers{false, true, false, true, true, true};
  for (std::size_t index = 0u; index < run.step_count; ++index) {
    const bool map = index < 3u;
    const auto kind = map ? rund::kernel::NodeKind::Map
                          : (index < 5u ? rund::kernel::NodeKind::Gather
                                        : rund::kernel::NodeKind::Reduce);
    const BoundStep &step = run.steps[index];
    // The count Map is independent of the preceding item-index Map. Graph
    // output visibility may conservatively retain the boundary between them,
    // while an internal-only count may omit it. Both schedules have the same
    // proved dataflow; all other boundaries are exact.
    const bool barrier_valid =
        index == 2u || step.barrier_before == barriers[index];
    if (!BoundStepMatches(step, kind) || step.step == nullptr ||
        step.planned == nullptr || step.source_binds == nullptr ||
        step.control.active() || !step.resets.empty() || !barrier_valid) {
      reason = "compute_pipeline_nested_aggregate_seed_schedule_ineligible";
      return false;
    }
  }

  BindingSet base{};
  BindingSet items{};
  BindingSet count{};
  if (!MapShape(run.steps[0u], MapSemanticKind::ResidentWindowBaseU32, 1u, 1u,
                base) ||
      !MapShape(run.steps[1u], MapSemanticKind::ResidentWindowItemsU32, 1u, 1u,
                items) ||
      !MapShape(run.steps[2u], MapSemanticKind::ResidentWindowCountU32, 2u, 1u,
                count) ||
      run.steps[0u].step->map_semantic.tile != window.tile ||
      run.steps[2u].step->map_semantic.maximum != window.maximum ||
      run.steps[2u].step->map_semantic.tile != window.tile ||
      run.steps[2u].step->map_semantic.windows != window.outer_bound) {
    reason = "compute_pipeline_nested_aggregate_seed_map_ineligible";
    return false;
  }

  const View ordinal = At(base.resident_inputs, 0u);
  const View base_value = At(base.resident_outputs, 0u);
  const View items_base = At(items.resident_inputs, 0u);
  const View item_values = At(items.resident_outputs, 0u);
  const View total = At(count.resident_inputs, 0u);
  const View count_ordinal = At(count.resident_inputs, 1u);
  const View active_count = At(count.resident_outputs, 0u);
  View queue{};
  View queue_indices{};
  View queue_count{};
  View gathered_queue{};
  View domain{};
  View domain_indices{};
  View domain_count{};
  View gathered_domain{};
  View reduce_input{};
  View reduce_count{};
  View tile_state{};
  if (!GatherShape(run.steps[3u], window.tile, queue, queue_indices,
                   queue_count, gathered_queue) ||
      !GatherShape(run.steps[4u], window.tile, domain, domain_indices,
                   domain_count, gathered_domain) ||
      !ReduceShape(run.steps[5u], window.tile, reduce_input, reduce_count,
                   tile_state) ||
      !ReadView(ordinal) || !WriteView(base_value) || !ReadView(items_base) ||
      !WriteView(item_values) || !ReadView(total) || !ReadView(count_ordinal) ||
      !WriteView(active_count) || !U32View(ordinal, 1u) ||
      !U32View(base_value, 1u) || !U32View(items_base, 1u) ||
      !U32View(item_values, window.tile) || !U32View(total, 1u) ||
      !U32View(count_ordinal, 1u) || !U32View(active_count, 1u) ||
      !U32View(queue, queue.ref->count) ||
      !U32View(domain, domain.ref->count) ||
      queue.ref->count < window.maximum || domain.ref->count == 0u ||
      !DenseU32Workspace(item_values, window.outer_bound) ||
      !DenseU32Workspace(gathered_queue, window.outer_bound) ||
      !InternalOutput(run, run.steps[1u]) ||
      !InternalOutput(run, run.steps[3u]) ||
      !DisjointStorage(item_values, gathered_queue) ||
      !SameStorage(ordinal, count_ordinal) ||
      !SameStorage(base_value, items_base) ||
      !SameStorage(item_values, queue_indices) ||
      !SameStorage(active_count, queue_count) ||
      !SameStorage(active_count, domain_count) ||
      !SameStorage(active_count, reduce_count) ||
      !SameStorage(gathered_queue, domain_indices) ||
      !SameStorage(gathered_domain, reduce_input) ||
      !SameStorage(total, At(&window.count.source, &window.count.handle))) {
    reason = "compute_pipeline_nested_aggregate_seed_lineage_ineligible";
    return false;
  }
  projection = SeedProjection{
      .queue = Read(queue),
      .domain = Read(domain),
      .count = Read(total),
      .tile_low = item_values,
      .tile_status = gathered_queue,
      .tile_state = tile_state,
      .tile_count = active_count,
      .invalid_index_source_node = run.steps[4u].step->source.begin.index,
      .reduce_overflow_source_node = run.steps[5u].step->source.begin.index,
  };
  reason = "ok";
  return true;
}

[[nodiscard]] bool
ExactTemplateShape(const std::span<const BackendBatchEntry> templates,
                   const std::span<const std::uint8_t> barriers,
                   const std::size_t first, NestedTemplateSpan &seed,
                   NestedTemplateSpan &action, NestedTemplateSpan &fold,
                   BackendWindow &window) noexcept {
  if (first >= templates.size() || barriers.size() != templates.size()) {
    return false;
  }
  const BackendWindow *const source = templates[first].recurrence.window;
  if (source == nullptr || source->phase != BackendWindowPhase::NestedSeed ||
      source->maximum == 0u || source->tile == 0u ||
      source->tile > source->maximum || source->outer_bound == 0u ||
      source->inner_bound == 0u || source->has_terminal ||
      source->tile == std::numeric_limits<std::uint32_t>::max() ||
      source->outer_iteration != 0u || source->route != 0u) {
    return false;
  }
  const std::uint64_t expected_outer =
      (static_cast<std::uint64_t>(source->maximum) + source->tile - 1u) /
      source->tile;
  const std::uint64_t group_count = expected_outer + source->inner_bound + 3u;
  if (expected_outer != source->outer_bound ||
      group_count > templates.size() - first ||
      first > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  const std::size_t action_first = first + source->outer_bound;
  const std::size_t fold_first = action_first + source->inner_bound;
  for (std::size_t index = first; index < first + group_count; ++index) {
    const BackendBatchEntry &entry = templates[index];
    const BackendWindow *const current = entry.recurrence.window;
    if (current == nullptr || !SameWindowIdentity(*source, *current) ||
        entry.run == nullptr || entry.template_index != index ||
        (index != first && barriers[index] == 0u)) {
      return false;
    }
    if (index < action_first) {
      const std::uint32_t outer = static_cast<std::uint32_t>(index - first);
      if (current->phase != BackendWindowPhase::NestedSeed ||
          current->outer_iteration != outer || current->route != 0u ||
          entry.recurrence.iteration != outer ||
          entry.recurrence.bound != source->outer_bound) {
        return false;
      }
    } else if (index < fold_first) {
      const std::uint32_t inner =
          static_cast<std::uint32_t>(index - action_first);
      if (current->phase != BackendWindowPhase::NestedAction ||
          current->inner_iteration != inner || current->route != 0u ||
          entry.recurrence.iteration != inner ||
          entry.recurrence.bound != source->inner_bound) {
        return false;
      }
    } else {
      const std::uint32_t route =
          static_cast<std::uint32_t>(index - fold_first);
      if (current->phase != BackendWindowPhase::NestedFold ||
          current->route != route || entry.recurrence.iteration != route ||
          entry.recurrence.bound != 3u) {
        return false;
      }
    }
    if (entry.recurrence.logical_step !=
        templates[first].recurrence.logical_step) {
      return false;
    }
  }
  seed = NestedTemplateSpan{.first = static_cast<std::uint32_t>(first),
                            .count = source->outer_bound};
  action = NestedTemplateSpan{
      .first = static_cast<std::uint32_t>(action_first),
      .count = source->inner_bound,
  };
  fold = NestedTemplateSpan{.first = static_cast<std::uint32_t>(fold_first),
                            .count = 3u};
  window = *source;
  return true;
}

[[nodiscard]] bool ScalarMap(const BackendRun &run,
                             const ProgramFingerprint &program,
                             const MapSemanticKind semantic,
                             const std::uint64_t inputs, BindingSet &bindings,
                             const std::uint64_t final_dispatches) noexcept {
  return ReadyRun(&run, 1u) && SameProgram(run, program) &&
         run.final_dispatch_count == final_dispatches &&
         ReadyStep(run.steps[0u], rund::kernel::NodeKind::Map, true) &&
         MapShape(run.steps[0u], semantic, inputs, 1u, bindings) &&
         run.steps[0u].step->element_count == 1u &&
         U32View(At(bindings.resident_outputs, 0u), 1u) &&
         WriteView(At(bindings.resident_outputs, 0u));
}

[[nodiscard]] bool
BuildAction(const std::span<const BackendBatchEntry> templates,
            const NestedTemplateSpan action, const SeedProjection &seed,
            ProgramFingerprint &program, NestedScalarExpr &expression,
            View &final_state, std::uint32_t &dispatches) noexcept {
  const BackendRun *const first = templates[action.first].run;
  if (!ReadyRun(first, 1u) || first->steps[0u].step == nullptr) {
    return false;
  }
  program = ProgramIdentity(*first);
  if (!U32Program(program) || first->final_dispatch_count == 0u ||
      first->final_dispatch_count > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  dispatches = static_cast<std::uint32_t>(first->final_dispatch_count);
  View previous = seed.tile_state;
  for (std::uint32_t iteration = 0u; iteration < action.count; ++iteration) {
    const BackendRun *const run = templates[action.first + iteration].run;
    if (run == nullptr || !ReadyRun(run, 1u) ||
        run->steps[0u].step == nullptr) {
      return false;
    }
    const MapSemantic &semantic = run->steps[0u].step->map_semantic;
    const bool immediate =
        semantic.kind == MapSemanticKind::AddWrapU32Immediate;
    const std::uint64_t input_count = immediate ? 1u : 2u;
    BindingSet bindings{};
    if (!ScalarMap(*run, program, semantic.kind, input_count, bindings,
                   dispatches) ||
        (!immediate && semantic.kind != MapSemanticKind::AddWrapU32Pair)) {
      return false;
    }
    const View input = At(bindings.resident_inputs, 0u);
    const View output = At(bindings.resident_outputs, 0u);
    if (!ReadView(input) || !U32View(input, 1u) ||
        !SameStorage(input, previous) || SameStorage(input, output) ||
        (!immediate &&
         (!ReadView(At(bindings.resident_inputs, 1u)) ||
          !U32View(At(bindings.resident_inputs, 1u), 1u) ||
          !SameStorage(At(bindings.resident_inputs, 1u), seed.tile_count)))) {
      return false;
    }
    const NestedScalarExpr current{
        .op = NestedScalarOp::AddWrapU32,
        .lhs = NestedScalarValue::TileState,
        .rhs = immediate ? NestedScalarValue::Immediate
                         : NestedScalarValue::TileCount,
        .immediate = semantic.immediate,
    };
    if (iteration == 0u) {
      expression = current;
    } else if (expression.op != current.op || expression.lhs != current.lhs ||
               expression.rhs != current.rhs ||
               expression.immediate != current.immediate) {
      return false;
    }
    previous = output;
  }
  final_state = previous;
  return true;
}

[[nodiscard]] bool
BuildFold(const std::span<const BackendBatchEntry> templates,
          const NestedTemplateSpan fold, const SeedProjection &seed,
          const View action_state, const BackendPublish &publication,
          ProgramFingerprint &program, NestedScalarExpr &expression,
          std::uint32_t &dispatches) noexcept {
  const BackendRun *const first = templates[fold.first].run;
  if (!ReadyRun(first, 1u) || first->steps[0u].step == nullptr) {
    return false;
  }
  program = ProgramIdentity(*first);
  if (!U32Program(program) || first->final_dispatch_count == 0u ||
      first->final_dispatch_count > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  dispatches = static_cast<std::uint32_t>(first->final_dispatch_count);
  std::array<View, 3u> inputs{};
  std::array<View, 3u> outputs{};
  for (std::uint32_t route = 0u; route < fold.count; ++route) {
    const BackendRun *const run = templates[fold.first + route].run;
    if (!ReadyRun(run, 1u) || run->steps[0u].step == nullptr) {
      return false;
    }
    const BindingSet candidate = MapBindingFor(run->steps[0u]);
    const std::uint64_t input_count = candidate.resident_inputs.count;
    BindingSet bindings{};
    if ((input_count != 2u && input_count != 3u) ||
        !ScalarMap(*run, program, MapSemanticKind::AddWrapU32Pair, input_count,
                   bindings, dispatches)) {
      return false;
    }
    if (!ReadView(At(bindings.resident_inputs, 0u)) ||
        !ReadView(At(bindings.resident_inputs, 1u)) ||
        !U32View(At(bindings.resident_inputs, 0u), 1u) ||
        !U32View(At(bindings.resident_inputs, 1u), 1u) ||
        !SameStorage(At(bindings.resident_inputs, 1u), action_state) ||
        (input_count == 3u &&
         (!ReadView(At(bindings.resident_inputs, 2u)) ||
          !U32View(At(bindings.resident_inputs, 2u), 1u) ||
          !SameStorage(At(bindings.resident_inputs, 2u), seed.tile_count)))) {
      return false;
    }
    inputs[route] = At(bindings.resident_inputs, 0u);
    outputs[route] = At(bindings.resident_outputs, 0u);
  }
  if (!SameStorage(inputs[0u], At(&publication.sources[0u].source,
                                  &publication.sources[0u].handle)) ||
      !SameStorage(outputs[0u], At(&publication.sources[1u].source,
                                   &publication.sources[1u].handle)) ||
      !SameStorage(inputs[1u], outputs[0u]) ||
      !SameStorage(outputs[1u], At(&publication.sources[2u].source,
                                   &publication.sources[2u].handle)) ||
      !SameStorage(inputs[2u], outputs[1u]) ||
      !SameStorage(outputs[2u], outputs[0u])) {
    return false;
  }
  expression = NestedScalarExpr{
      .op = NestedScalarOp::AddWrapU32,
      .lhs = NestedScalarValue::OuterState,
      .rhs = NestedScalarValue::TileState,
  };
  return true;
}

[[nodiscard]] bool
PublicationFor(const std::span<const BackendPublish> publications,
               const BackendWindow &window, BackendPublish &publication,
               std::uint32_t &index) noexcept {
  std::uint32_t found = NoNode;
  for (std::size_t current = 0u; current < publications.size(); ++current) {
    if (publications[current].state != window.state) {
      continue;
    }
    if (found != NoNode || current >= NoNode) {
      return false;
    }
    found = static_cast<std::uint32_t>(current);
  }
  if (found == NoNode) {
    return false;
  }
  const BackendPublish &source = publications[found];
  const std::uint32_t expected_final = 1u + ((window.outer_bound - 1u) & 1u);
  const View target = At(&source.target, &source.target_handle);
  if (source.final != expected_final || !WriteView(target) ||
      !U32View(target, 1u)) {
    return false;
  }
  for (const BackendRead &read : source.sources) {
    const View view = At(&read.source, &read.handle);
    if (!ReadView(view) || !U32View(view, 1u)) {
      return false;
    }
  }
  publication = source;
  index = found;
  return true;
}

} // namespace

NestedAggregate
BuildNestedAggregate(const std::span<const BackendBatchEntry> templates,
                     const std::span<const std::uint8_t> barriers,
                     const std::span<const BackendPublish> publications,
                     const std::size_t first) {
  NestedTemplateSpan seed{};
  NestedTemplateSpan action{};
  NestedTemplateSpan fold{};
  BackendWindow window{};
  if (barriers.size() != templates.size()) {
    return Invalid("compute_pipeline_nested_aggregate_barrier_invalid");
  }
  if (!ExactTemplateShape(templates, barriers, first, seed, action, fold,
                          window)) {
    return Ineligible("compute_pipeline_nested_aggregate_shape_ineligible");
  }

  const BackendRun *const seed_run = templates[seed.first].run;
  if (seed_run == nullptr) {
    return Ineligible("compute_pipeline_nested_aggregate_seed_ineligible");
  }
  const ProgramFingerprint seed_program = ProgramIdentity(*seed_run);
  if (!U32Program(seed_program)) {
    return Ineligible("compute_pipeline_nested_aggregate_seed_ineligible");
  }
  SeedProjection seed_projection{};
  const char *seed_reason = "compute_pipeline_nested_aggregate_seed_ineligible";
  for (std::uint32_t outer = 0u; outer < seed.count; ++outer) {
    const BackendRun *const run = templates[seed.first + outer].run;
    SeedProjection current{};
    if (run == nullptr || !SameProgram(*run, seed_program)) {
      return Ineligible(
          "compute_pipeline_nested_aggregate_seed_identity_ineligible");
    }
    if (!SeedShape(*run, window, current, seed_reason)) {
      return Ineligible(seed_reason);
    }
    if (outer != 0u &&
        (!SameRead(seed_projection.queue, current.queue) ||
         !SameRead(seed_projection.domain, current.domain) ||
         !SameRead(seed_projection.count, current.count) ||
         !SameStorage(seed_projection.tile_low, current.tile_low) ||
         !SameStorage(seed_projection.tile_status, current.tile_status) ||
         !SameStorage(seed_projection.tile_state, current.tile_state) ||
         !SameStorage(seed_projection.tile_count, current.tile_count) ||
         seed_projection.invalid_index_source_node !=
             current.invalid_index_source_node ||
         seed_projection.reduce_overflow_source_node !=
             current.reduce_overflow_source_node)) {
      return Ineligible(
          "compute_pipeline_nested_aggregate_seed_identity_ineligible");
    }
    if (outer == 0u) {
      seed_projection = std::move(current);
    }
  }

  ProgramFingerprint action_program{};
  NestedScalarExpr action_expression{};
  View action_state{};
  std::uint32_t action_dispatches{};
  if (!BuildAction(templates, action, seed_projection, action_program,
                   action_expression, action_state, action_dispatches)) {
    return Ineligible("compute_pipeline_nested_aggregate_action_ineligible");
  }

  BackendPublish publication{};
  std::uint32_t publication_index = NoNode;
  if (!PublicationFor(publications, window, publication, publication_index)) {
    return Ineligible(
        "compute_pipeline_nested_aggregate_publication_ineligible");
  }
  ProgramFingerprint fold_program{};
  NestedScalarExpr fold_expression{};
  std::uint32_t fold_dispatches{};
  if (!BuildFold(templates, fold, seed_projection, action_state, publication,
                 fold_program, fold_expression, fold_dispatches)) {
    return Ineligible("compute_pipeline_nested_aggregate_fold_ineligible");
  }

  const std::uint64_t seed_dispatches = seed_run->final_dispatch_count;
  if (seed_dispatches == 0u ||
      seed_dispatches > std::numeric_limits<std::uint32_t>::max()) {
    return Ineligible("compute_pipeline_nested_aggregate_profile_ineligible");
  }
  const std::uint64_t authored_action =
      static_cast<std::uint64_t>(window.outer_bound) * window.inner_bound;
  return NestedAggregate{
      .state = NestedAggregateState::Ready,
      .seed = seed,
      .action = action,
      .fold = fold,
      .window = window,
      .queue =
          NestedAggregateRead{
              .read = std::move(seed_projection.queue),
              .logical_count = seed_projection.queue.source.count,
              .element_bytes = seed_projection.queue.source.element_bytes,
          },
      .domain =
          NestedAggregateRead{
              .read = std::move(seed_projection.domain),
              .logical_count = seed_projection.domain.source.count,
              .element_bytes = seed_projection.domain.source.element_bytes,
          },
      .count =
          NestedAggregateRead{
              .read = std::move(seed_projection.count),
              .logical_count = 1u,
              .element_bytes = sizeof(std::uint32_t),
          },
      .tile_low = Workspace(seed_projection.tile_low),
      .tile_status = Workspace(seed_projection.tile_status),
      .action_expr = action_expression,
      .fold_expr = fold_expression,
      .publication = std::move(publication),
      .publication_index = publication_index,
      .failure =
          NestedAggregateFailureProjection{
              .logical_step = templates[seed.first].recurrence.logical_step,
              .invalid_index_source_node =
                  seed_projection.invalid_index_source_node,
              .reduce_overflow_source_node =
                  seed_projection.reduce_overflow_source_node,
              .count_overflow_reason = static_cast<std::uint32_t>(
                  rund::compute::Reason::BoundedCountInvalid),
              .invalid_index_reason = static_cast<std::uint32_t>(
                  rund::compute::Reason::GatherIndexOutOfRange),
              .reduce_overflow_reason = static_cast<std::uint32_t>(
                  rund::compute::Reason::ReduceSumOverflow),
              .phase = rund::compute::PipelineNestedPhase::Seed,
              .inner_coordinate_unknown = true,
          },
      .profile =
          NestedAggregateProfileProjection{
              .authored_seed_occurrences = window.outer_bound,
              .authored_action_occurrences = authored_action,
              .authored_fold_occurrences = window.outer_bound,
              .seed_dispatches_per_occurrence =
                  static_cast<std::uint32_t>(seed_dispatches),
              .action_dispatches_per_occurrence = action_dispatches,
              .fold_dispatches_per_occurrence = fold_dispatches,
              .aggregate_profile_supported = true,
          },
      .reason = "ok",
  };
}

} // namespace rund::node::accel::detail
