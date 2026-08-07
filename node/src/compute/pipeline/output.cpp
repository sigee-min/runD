#include "output.hpp"

#include "../program/output.hpp"

#include <cstddef>
#include <limits>
#include <utility>

namespace rund::compute::detail {

Result<OutputProjection> project_outputs(const ProgramState &program,
                                         const std::size_t logical_outputs) {
  const std::size_t physical = program.output_types.size();
  const std::size_t logical = output_count(program.output_aliases, physical);
  if (logical > PipelineLeafCapacity || physical > PipelineLeafCapacity) {
    return Result<OutputProjection>::fail(Reason::PipelineCapacity);
  }
  if (physical == 0u || logical_outputs != logical ||
      program.output_sizes.size() != physical ||
      program.output_formats.size() != physical) {
    return Result<OutputProjection>::fail(Reason::BindingCountMismatch);
  }
  OutputProjection projection{};
  projection.physical_sources.fill(OutputProjection::unassigned);
  projection.physical_count = physical;
  std::size_t assigned = 0u;
  for (std::size_t index = 0u; index < logical; ++index) {
    const std::size_t target = output_index(program.output_aliases, index);
    if (target >= physical) {
      return Result<OutputProjection>::fail(Reason::GraphBindingInvalid);
    }
    projection.logical_to_physical[index] = static_cast<std::uint32_t>(target);
    std::uint32_t &source = projection.physical_sources[target];
    if (source != OutputProjection::unassigned) {
      continue;
    }
    source = static_cast<std::uint32_t>(index);
    ++assigned;
  }
  if (assigned != physical) {
    return Result<OutputProjection>::fail(Reason::BindingOutputMissing);
  }
  return Result<OutputProjection>::success(std::move(projection));
}

Result<OutputProjection> project_outputs(const PipelineBuildStep &step) {
  if (step.program == nullptr) {
    return Result<OutputProjection>::fail(Reason::PipelineInvalid);
  }
  return project_outputs(*step.program, step.outputs.size());
}

Result<PipelineBuildOutputProjection>
resolve_build_output(const PipelineBuildState &build,
                     const PipelineBuildOutputCoordinate coordinate) {
  if (coordinate.step.value >= build.steps.size()) {
    return Result<PipelineBuildOutputProjection>::fail(Reason::PipelineInvalid);
  }
  const PipelineBuildStep &step = build.steps[coordinate.step.value];
  auto projection = project_outputs(step);
  if (!projection || coordinate.output.value >= step.outputs.size()) {
    return Result<PipelineBuildOutputProjection>::fail(
        projection ? Reason::PipelineInvalid : projection.reason());
  }
  const std::uint32_t physical =
      projection->logical_to_physical[coordinate.output.value];
  if (physical >= projection->physical_count ||
      physical >= projection->physical_sources.size()) {
    return Result<PipelineBuildOutputProjection>::fail(Reason::PipelineInvalid);
  }
  const std::uint32_t source = projection->physical_sources[physical];
  if (source == OutputProjection::unassigned || source >= step.outputs.size()) {
    return Result<PipelineBuildOutputProjection>::fail(Reason::PipelineInvalid);
  }
  return Result<PipelineBuildOutputProjection>::success({
      .physical = {.value = physical},
      .source = {.value = source},
  });
}

Result<PipelineBuildWindowFinal>
resolve_build_window_final(const PipelineBuildState &build,
                           const PipelineBuildWindowControlOrdinal ordinal) {
  auto anchors = resolve_build_window_anchors(build, ordinal);
  if (!anchors) {
    return Result<PipelineBuildWindowFinal>::fail(anchors.reason());
  }
  const PipelineBuildWindowControl &control =
      build.window_controls[ordinal.value];
  const PipelineBuildStep &count_step = build.steps[anchors->count_step.value];
  const PipelineBuildStep &terminal_step =
      build.steps[anchors->terminal_step.value];
  std::size_t bound = count_step.iteration_bound;
  std::size_t source_step = anchors->terminal_step.value;
  std::uint32_t bank = PipelineWindow::seed;
  if (control.nested == 0u) {
    if (count_step.route != PipelineRoute::Ordinary ||
        terminal_step.route != PipelineRoute::Ordinary ||
        count_step.nested != 0u || terminal_step.nested != 0u ||
        count_step.iteration != 0u || bound == 0u ||
        bound > build.steps.size() - source_step) {
      return Result<PipelineBuildWindowFinal>::fail(Reason::PipelineInvalid);
    }
    source_step += bound - 1u;
    bank =
        static_cast<std::uint32_t>(PipelineWindow::first + ((bound - 1u) & 1u));
  } else {
    const PipelineBuildNestedWindow &nested =
        build.nested_windows[control.nested - 1u];
    bound = nested.shape.outer_bound();
    if (bound == 0u || count_step.nested != control.nested ||
        terminal_step.nested != control.nested ||
        terminal_step.route != PipelineRoute::NestedFold ||
        build.steps.size() < 3u ||
        nested.shape.fold_first() > build.steps.size() - 3u) {
      return Result<PipelineBuildWindowFinal>::fail(Reason::PipelineInvalid);
    }
    std::uint32_t route = 0u;
    if (!nested.shape.fold_route_for_outer(
            static_cast<std::uint32_t>(bound - 1u), route)) {
      return Result<PipelineBuildWindowFinal>::fail(Reason::PipelineInvalid);
    }
    source_step += route;
    bank = route == 1u ? PipelineWindow::second : PipelineWindow::first;
  }
  if (bound > std::numeric_limits<std::uint32_t>::max() ||
      source_step >= build.steps.size()) {
    return Result<PipelineBuildWindowFinal>::fail(Reason::PipelineInvalid);
  }
  return Result<PipelineBuildWindowFinal>::success({
      .source_step = {.value = source_step},
      .bank = bank,
  });
}

Result<PipelineBuildWindowAnchors>
resolve_build_window_anchors(const PipelineBuildState &build,
                             const PipelineBuildWindowControlOrdinal control) {
  if (control.value >= build.window_controls.size()) {
    return Result<PipelineBuildWindowAnchors>::fail(Reason::PipelineInvalid);
  }
  const PipelineBuildWindowControl &authored =
      build.window_controls[control.value];
  if (authored.nested == 0u) {
    if (authored.ordinary_step.value >= build.steps.size()) {
      return Result<PipelineBuildWindowAnchors>::fail(Reason::PipelineInvalid);
    }
    return Result<PipelineBuildWindowAnchors>::success({
        .count_step = authored.ordinary_step,
        .terminal_step = authored.ordinary_step,
    });
  }
  const std::size_t nested_index = authored.nested - 1u;
  if (nested_index >= build.nested_windows.size()) {
    return Result<PipelineBuildWindowAnchors>::fail(Reason::PipelineInvalid);
  }
  const PipelineBuildNestedWindow &nested = build.nested_windows[nested_index];
  if (nested.shape.seed_first() >= build.steps.size() ||
      nested.shape.fold_first() >= build.steps.size()) {
    return Result<PipelineBuildWindowAnchors>::fail(Reason::PipelineInvalid);
  }
  return Result<PipelineBuildWindowAnchors>::success({
      .count_step = {.value = nested.shape.seed_first()},
      .terminal_step = {.value = nested.shape.fold_first()},
  });
}

Result<PipelineBuildPublicationBase>
resolve_publication_base(const PipelineBuildState &build,
                         const PipelineBuildPublicationEdge &edge) {
  if (edge.control.value >= build.window_controls.size()) {
    return Result<PipelineBuildPublicationBase>::fail(Reason::PipelineInvalid);
  }
  auto anchors = resolve_build_window_anchors(build, edge.control);
  if (!anchors ||
      build.steps[anchors->terminal_step.value].window_control.value !=
          edge.control.value) {
    return Result<PipelineBuildPublicationBase>::fail(Reason::PipelineInvalid);
  }
  return Result<PipelineBuildPublicationBase>::success({
      .control = edge.control,
      .step = anchors->terminal_step,
  });
}

Result<PipelineBuildOutputCoordinate> resolve_publication_source(
    const PipelineBuildState &build,
    const PipelineBuildTerminalPublication &publication) {
  auto base = resolve_publication_base(build, publication.edge);
  if (!base) {
    return Result<PipelineBuildOutputCoordinate>::fail(base.reason());
  }
  auto selected = resolve_build_window_final(build, base->control);
  if (!selected) {
    return Result<PipelineBuildOutputCoordinate>::fail(selected.reason());
  }
  return Result<PipelineBuildOutputCoordinate>::success({
      .step = selected->source_step,
      .output = publication.edge.output,
  });
}

Result<PipelineBuildOutputCoordinate>
resolve_publication_source(const PipelineBuildState &build,
                           const PipelineBuildWindowPublication &publication) {
  auto base = resolve_publication_base(build, publication.edge);
  if (!base || build.window_controls[base->control.value].nested == 0u) {
    return Result<PipelineBuildOutputCoordinate>::fail(
        base ? Reason::PipelineInvalid : base.reason());
  }
  auto selected = resolve_build_window_final(build, base->control);
  if (!selected) {
    return Result<PipelineBuildOutputCoordinate>::fail(selected.reason());
  }
  return Result<PipelineBuildOutputCoordinate>::success({
      .step = base->step,
      .output = publication.edge.output,
  });
}

} // namespace rund::compute::detail
