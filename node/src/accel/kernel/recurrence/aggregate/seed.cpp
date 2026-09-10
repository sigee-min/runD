#include "internal.hpp"

#include <kernel/program/compute/gather/plan.hpp>
#include <kernel/program/compute/reduce/plan.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

namespace rund::node::accel::detail::nested_aggregate_detail {

bool MapShape(const BoundStep &step, const MapSemanticKind semantic,
              const std::uint64_t inputs, const std::uint64_t outputs,
              BindingSet &bindings) noexcept {
  if (!BoundStepMatches(step, rund::kernel::NodeKind::Map) ||
      step.step == nullptr || step.planned == nullptr ||
      step.source_binds == nullptr || step.control.active() ||
      !step.resets.empty() || step.step->map_semantic.kind != semantic ||
      !step.step->artifact.ok || !step.step->graph_binding_indices_ok ||
      !step.planned->plan.ok || step.planned->artifact == nullptr ||
      step.planned->artifact != &step.step->artifact ||
      step.step->artifact.key.scalar != rund::kernel::ComputeScalar::Lane32 ||
      step.step->artifact.key.domain != rund::kernel::ComputeDomain::U32) {
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

bool CollectiveStep(const BoundStep &step,
                    const rund::kernel::NodeKind kind) noexcept {
  return ReadyStep(step, kind, false) && step.step->source.valid();
}

bool GatherShape(const BoundStep &step, const std::uint32_t tile,
                 View &values, View &indices, View &count,
                 View &output) noexcept {
  const operation::Gather *const operation =
      OperationFor<operation::Gather>(step);
  const GatherBinds *const bindings = BindingsFor<GatherBinds>(step);
  if (!CollectiveStep(step, rund::kernel::NodeKind::Gather) ||
      operation == nullptr || bindings == nullptr ||
      operation->desc.element != rund::kernel::GatherElement::U32 ||
      operation->desc.element_count != tile ||
      operation->desc.count_source != rund::kernel::ComputeCountSource::BufferU32 ||
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

bool ReduceShape(const BoundStep &step, const std::uint32_t tile,
                 View &input, View &count, View &output) noexcept {
  const operation::Reduce *const operation =
      OperationFor<operation::Reduce>(step);
  const ReduceBinds *const bindings = BindingsFor<ReduceBinds>(step);
  if (!CollectiveStep(step, rund::kernel::NodeKind::Reduce) ||
      operation == nullptr || bindings == nullptr ||
      operation->desc.op != rund::kernel::ReduceOp::Sum ||
      operation->desc.element != rund::kernel::ReduceElement::U32 ||
      operation->desc.element_count != tile ||
      operation->desc.count_source != rund::kernel::ComputeCountSource::BufferU32 ||
      !rund::kernel::ReducePlanMatchesDesc(operation->desc, operation->plan)) {
    return false;
  }
  input = At(bindings->input, bindings->input_handle);
  count = At(bindings->logical_count, bindings->logical_count_handle);
  output = At(bindings->output, bindings->output_handle);
  return ReadView(input) && ReadView(count) && WriteView(output) &&
         U32View(input, tile) && U32View(count, 1u) && U32View(output, 1u);
}

bool SeedShape(const BackendRun &run, const BackendWindow &window,
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
  if (!MapShape(run.steps[0u], MapSemanticKind::MulWrapU32Immediate, 1u, 1u,
                base) ||
      !MapShape(run.steps[1u], MapSemanticKind::ResidentWindowItemsU32, 1u, 1u,
                items) ||
      !MapShape(run.steps[2u], MapSemanticKind::ResidentWindowCountU32, 2u, 1u,
                count) ||
      run.steps[0u].step->map_semantic.immediate != window.tile ||
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

} // namespace rund::node::accel::detail::nested_aggregate_detail
