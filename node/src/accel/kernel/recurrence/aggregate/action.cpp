#include "internal.hpp"

#include <limits>

namespace rund::node::accel::detail::nested_aggregate_detail {

bool ExactAggregateEnvelope(
    const std::span<const BackendBatchEntry> templates,
    const std::span<const std::uint8_t> barriers, const std::size_t first,
    NestedTemplateGeometry &geometry) noexcept {
  if (!ProveNestedTemplateGeometry(templates, first, geometry) ||
      geometry.window() == nullptr || geometry.inner_bound() == 0u ||
      geometry.window()->has_terminal ||
      geometry.window()->tile ==
          std::numeric_limits<std::uint32_t>::max() ||
      first > std::numeric_limits<std::uint32_t>::max() ||
      geometry.action_first() > std::numeric_limits<std::uint32_t>::max() ||
      geometry.fold_first() > std::numeric_limits<std::uint32_t>::max() ||
      geometry.end() > static_cast<std::uint64_t>(
                           std::numeric_limits<std::uint32_t>::max()) +
                           1u) {
    return false;
  }
  for (std::size_t index = first; index < geometry.end(); ++index) {
    const BackendBatchEntry &entry = templates[index];
    if (entry.run == nullptr || entry.template_index != index ||
        (index != first && barriers[index] == 0u)) {
      return false;
    }
  }
  return true;
}

bool ScalarMap(const BackendRun &run, const ProgramFingerprint &program,
               const MapSemanticKind semantic, const std::uint64_t inputs,
               BindingSet &bindings,
               const std::uint64_t final_dispatches) noexcept {
  return ReadyRun(&run, 1u) && SameProgram(run, program) &&
         run.final_dispatch_count == final_dispatches &&
         ReadyStep(run.steps[0u], rund::kernel::NodeKind::Map, true) &&
         MapShape(run.steps[0u], semantic, inputs, 1u, bindings) &&
         run.steps[0u].step->element_count == 1u &&
         U32View(At(bindings.resident_outputs, 0u), 1u) &&
         WriteView(At(bindings.resident_outputs, 0u));
}

bool BuildAction(const std::span<const BackendBatchEntry> templates,
                 const NestedTemplateShape &shape, const SeedProjection &seed,
                 ProgramFingerprint &program, NestedScalarExpr &expression,
                 View &final_state, std::uint32_t &dispatches) noexcept {
  const BackendRun *const first = templates[shape.action_first()].run;
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
  for (std::uint32_t iteration = 0u; iteration < shape.action_count();
       ++iteration) {
    const BackendRun *const run =
        templates[shape.action_first() + iteration].run;
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

} // namespace rund::node::accel::detail::nested_aggregate_detail
