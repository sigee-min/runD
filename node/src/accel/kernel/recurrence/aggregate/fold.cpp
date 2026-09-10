#include "internal.hpp"

#include <array>

namespace rund::node::accel::detail::nested_aggregate_detail {

bool BuildFold(const std::span<const BackendBatchEntry> templates,
               const NestedTemplateShape &shape, const SeedProjection &seed,
               const View action_state, const BackendPublish &publication,
               ProgramFingerprint &program, NestedScalarExpr &expression,
               std::uint32_t &dispatches) noexcept {
  const BackendRun *const first = templates[shape.fold_first()].run;
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
  for (std::uint32_t route = 0u; route < shape.fold_count(); ++route) {
    const BackendRun *const run = templates[shape.fold_first() + route].run;
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
  if (!SameStorage(inputs[0u],
                  At(&publication.sources[0u].source,
                     &publication.sources[0u].handle)) ||
      !SameStorage(outputs[0u],
                   At(&publication.sources[1u].source,
                      &publication.sources[1u].handle)) ||
      !SameStorage(inputs[1u], outputs[0u]) ||
      !SameStorage(outputs[1u],
                   At(&publication.sources[2u].source,
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

bool PublicationFor(const std::span<const BackendPublish> publications,
                    const BackendWindow &window, BackendPublish &publication,
                    std::uint32_t &index) noexcept {
  std::uint32_t found = NoNode;
  for (std::size_t current = 0u; current < publications.size(); ++current) {
    if (publications[current].identity.state != window.state ||
        publications[current].identity.kind !=
            PreparedKernelPublicationKind::Terminal) {
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
  const View target = At(&source.target.source, &source.target.handle);
  if (source.identity.final >= source.sources.size() || !WriteView(target) ||
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

} // namespace rund::node::accel::detail::nested_aggregate_detail
