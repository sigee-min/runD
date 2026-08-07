#include "../local.hpp"
#include "../state.hpp"

#include "../../cpu/graph.hpp"
#include "../../cpu/view.hpp"
#include "../../job/local.hpp"
#include "../../status.hpp"

#include <rund/counter.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>

namespace rund::compute::detail {
namespace {

[[nodiscard]] PipelineWindow *window(PipelineState &state,
                                     const PipelineStep &step) noexcept {
  const std::size_t index = step.window;
  return index == 0u || index > state.windows.size()
             ? nullptr
             : &state.windows[index - 1u];
}

[[nodiscard]] Status window_status(PipelineState &state,
                                   const PipelineStep &step,
                                   const Status status) noexcept {
  const PipelineWindow *const descriptor = window(state, step);
  if (descriptor == nullptr || status.reason() != Reason::BoundedCountInvalid) {
    return status;
  }
  state.stats.control.overflow_ordinal = descriptor->control.maximum;
  return status;
}

[[nodiscard]] Status seal_resident(PipelineState &state,
                                   const PipelineWindow &window) noexcept {
  if (window.first_step >= state.steps.size() ||
      window.current > PipelineWindow::second ||
      window.recurrent_output_count == 0u) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::uint32_t final = window.control.final;
  if (final < PipelineWindow::first || final > PipelineWindow::second) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::uint32_t sealed_count = 0u;
  const std::uint32_t state_index =
      static_cast<std::uint32_t>(&window - state.windows.data());
  for (const PipelinePublicationPlan &publication : state.publications) {
    const auto *terminal =
        std::get_if<PipelineTerminalPublicationPlan>(&publication);
    if (terminal == nullptr || terminal->state != state_index) {
      continue;
    }
    if (window.current == final) {
      ++sealed_count;
      continue;
    }
    CpuView source{};
    CpuView target{};
    const Status source_ready = resolve_cpu_pipeline_publication_view(
        state, terminal->sources[window.current], source);
    const Status target_ready = resolve_cpu_pipeline_publication_view(
        state, terminal->sources[final], target);
    if (!source_ready || !target_ready ||
        source.footprint.bytes != target.footprint.bytes ||
        source.footprint.stride != target.footprint.stride ||
        source.footprint.width != target.footprint.width ||
        source.data == nullptr || target.data == nullptr) {
      return Status::fail(Reason::PipelineInvalid);
    }
    if (source.footprint.dense()) {
      std::memmove(target.data, source.data, source.footprint.bytes);
      ++sealed_count;
      continue;
    }
    const std::byte *read = source.data;
    std::byte *write = target.data;
    for (std::size_t remaining = source.footprint.count; remaining > 1u;
         --remaining) {
      std::memmove(write, read, source.footprint.width);
      read += source.footprint.stride;
      write += target.footprint.stride;
    }
    if (source.footprint.count != 0u) {
      std::memmove(write, read, source.footprint.width);
    }
    ++sealed_count;
  }
  return sealed_count == window.recurrent_output_count
             ? Status::success()
             : Status::fail(Reason::PipelineInvalid);
}

void accumulate(Stats &total, const Stats &value,
                const std::uint64_t conflict_count,
                const std::uint64_t overflow_ordinal,
                const bool control) noexcept {
  ::rund::detail::counter::Accumulate(total.dispatches, value.dispatches);
  ::rund::detail::counter::Accumulate(total.internal_roundtrip_bytes,
                                      value.internal_roundtrip_bytes);
  ::rund::detail::counter::Accumulate(total.reset_bytes, value.reset_bytes);
  ::rund::detail::counter::Accumulate(total.reset_commands,
                                      value.reset_commands);
  ::rund::detail::counter::Accumulate(total.graph_read_bytes,
                                      value.graph_read_bytes);
  total.worker_count = std::max(total.worker_count, value.worker_count);
  total.participating_workers =
      std::max(total.participating_workers, value.participating_workers);
  ::rund::detail::counter::Accumulate(total.tile_count, value.tile_count);
  total.tile_size = std::max(total.tile_size, value.tile_size);
  ::rund::detail::counter::Accumulate(total.vector_chunks, value.vector_chunks);
  ::rund::detail::counter::Accumulate(total.tail_chunks, value.tail_chunks);
  if (!control) {
    return;
  }
  ::rund::detail::counter::Accumulate(total.control.generated_item_count,
                                      value.control.generated_item_count);
  ::rund::detail::counter::Accumulate(total.control.generated_capacity,
                                      value.control.generated_capacity);
  ::rund::detail::counter::Accumulate(total.control.indirect_dispatch_count,
                                      value.control.indirect_dispatch_count);
  ::rund::detail::counter::Accumulate(total.control.indirect_work_item_count,
                                      value.control.indirect_work_item_count);
  ::rund::detail::counter::Accumulate(total.control.iteration_count,
                                      value.control.iteration_count);
  ::rund::detail::counter::Accumulate(total.control.skipped_iteration_count,
                                      value.control.skipped_iteration_count);
  ::rund::detail::counter::Accumulate(total.control.conflict_count,
                                      conflict_count);
  total.control.overflow_ordinal =
      std::min(total.control.overflow_ordinal, overflow_ordinal);
}

} // namespace

Status resolve_cpu_pipeline_publication_view(
    PipelineState &state, const PipelinePublicationViewPlan &planned,
    CpuView &view) noexcept {
  const PipelinePublicationViewIdentity &identity = planned.identity;
  const bool alternate = state.transactional && state.attempt_parity != 0u;
  const PipelineResource *const resource =
      selected_pipeline_resource(state, identity.resource_ordinal, alternate);
  if (resource == nullptr || resource->buffer == nullptr ||
      resource->type != planned.type || resource->format != planned.format ||
      resource->buffer->type != planned.type ||
      resource->bytes != identity.backing_bytes ||
      identity.element_bytes == 0u ||
      identity.offset_bytes % identity.element_bytes != 0u ||
      identity.stride_bytes % identity.element_bytes != 0u ||
      identity.offset_bytes / identity.element_bytes >
          std::numeric_limits<std::size_t>::max() ||
      identity.count > std::numeric_limits<std::size_t>::max() ||
      identity.stride_bytes / identity.element_bytes >
          std::numeric_limits<std::size_t>::max()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::optional<CpuView> resolved = cpu_view(
      resource->buffer.get(),
      static_cast<std::size_t>(identity.offset_bytes / identity.element_bytes),
      static_cast<std::size_t>(identity.count),
      static_cast<std::size_t>(identity.stride_bytes / identity.element_bytes),
      static_cast<std::size_t>(identity.element_bytes));
  if (!resolved || (identity.count != 0u && resolved->data == nullptr) ||
      resolved->footprint.base != identity.offset_bytes ||
      resolved->footprint.count != identity.count ||
      resolved->footprint.stride != identity.stride_bytes ||
      resolved->footprint.width != identity.element_bytes) {
    return Status::fail(Reason::PipelineInvalid);
  }
  view = *resolved;
  return Status::success();
}

void reset_cpu_resident(PipelineState &state) noexcept {
  for (PipelineWindow &window : state.windows) {
    window.current = PipelineWindow::seed;
    window.stopped = false;
  }
}

Status prepare_cpu_pipeline_window(PipelineState &state,
                                   const PipelineStep &step,
                                   bool &active) noexcept {
  active = true;
  if (step.window == 0u) {
    return Status::success();
  }
  PipelineWindow *const descriptor = window(state, step);
  if (descriptor == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::uint32_t item_count{};
  CpuView count{};
  const Status count_ready = resolve_cpu_pipeline_publication_view(
      state, descriptor->control.count, count);
  if (!count_ready || count.data == nullptr || count.footprint.count != 1u ||
      count.footprint.width != sizeof(item_count)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::memcpy(&item_count, count.data, sizeof(item_count));
  if (item_count > descriptor->control.maximum) {
    state.stats.control.overflow_ordinal = descriptor->control.maximum;
    return Status::fail(Reason::BoundedCountInvalid);
  }
  const std::uint64_t base =
      static_cast<std::uint64_t>(step.iteration) * descriptor->control.tile;
  bool terminal = false;
  if (descriptor->control.terminal_publication !=
      std::numeric_limits<std::uint32_t>::max()) {
    if (descriptor->control.terminal_publication >= state.publications.size()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const auto *terminal_plan = std::get_if<PipelineTerminalPublicationPlan>(
        &state.publications[descriptor->control.terminal_publication]);
    const std::uint32_t state_index =
        static_cast<std::uint32_t>(descriptor - state.windows.data());
    if (terminal_plan == nullptr || terminal_plan->state != state_index ||
        descriptor->current > PipelineWindow::second) {
      return Status::fail(Reason::PipelineInvalid);
    }
    CpuView view{};
    const Status terminal_ready = resolve_cpu_pipeline_publication_view(
        state, terminal_plan->sources[descriptor->current], view);
    if (!terminal_ready || view.footprint.count != 1u ||
        view.footprint.width != sizeof(std::uint32_t)) {
      return Status::fail(Reason::PipelineInvalid);
    }
    std::uint32_t observed{};
    std::memcpy(&observed, view.data, sizeof(observed));
    terminal = observed == descriptor->control.expected;
  }
  if (!descriptor->stopped && base < item_count && !terminal) {
    return Status::success();
  }

  active = false;
  if (!descriptor->stopped) {
    const Status sealed = seal_resident(state, *descriptor);
    if (!sealed) {
      return sealed;
    }
    descriptor->stopped = true;
  }
  ::rund::detail::counter::Accumulate(
      state.stats.control.skipped_iteration_count, 1u);
  return Status::success();
}

Status prepare_cpu_pipeline_window(PipelineState &state,
                                   const std::size_t index,
                                   bool &active) noexcept {
  if (index >= state.steps.size()) {
    active = false;
    return Status::fail(Reason::PipelineInvalid);
  }
  return prepare_cpu_pipeline_window(state, state.steps[index], active);
}

Status consume_cpu_pipeline_step(PipelineState &state, const std::size_t index,
                                 const Status execution) noexcept {
  if (index >= state.steps.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const PipelineStep &step = state.steps[index];
  const std::shared_ptr<JobState> &job =
      state.transactional && state.attempt_parity != 0u ? step.alternate_job
                                                        : step.job;
  if (step.program == nullptr || job == nullptr) {
    return Status::fail(Reason::RunInvalid);
  }
  if (step.program->empty()) {
    ::rund::detail::counter::Accumulate(state.stats.graph_read_bytes,
                                        step.program->graph_info.read_bytes);
    return Status::success();
  }
  if (job->cpu == nullptr || job->cpu->graph == nullptr) {
    return Status::fail(Reason::CpuRunInvalid);
  }
  const CpuGraphRun &graph = *job->cpu->graph;
  const Status semantic =
      graph.semantic_failure_count == 0u || graph.semantic_status == 0u
          ? Status::success()
          : Status::fail(primitive_execution_reason(graph.semantic_primitive,
                                                    graph.semantic_status));
  const Status primary = execution ? semantic : execution;
  const bool resident_overflow =
      step.window != 0u && primary.reason() == Reason::BoundedCountInvalid;
  const Status resolved = window_status(state, step, primary);
  accumulate(state.stats, job->cpu->stats, graph.conflict_count,
             graph.overflow_ordinal, !resident_overflow);
  if (resolved && step.window != 0u && step.route == PipelineRoute::Ordinary) {
    PipelineWindow *const descriptor = step.window <= state.windows.size()
                                           ? &state.windows[step.window - 1u]
                                           : nullptr;
    if (descriptor == nullptr || descriptor->stopped) {
      return Status::fail(Reason::PipelineInvalid);
    }
    descriptor->current = (step.iteration & 1u) == 0u ? PipelineWindow::first
                                                      : PipelineWindow::second;
    ::rund::detail::counter::Accumulate(state.stats.control.iteration_count,
                                        1u);
  }
  return resolved;
}

std::uint64_t cpu_program_status_entries(const ProgramState &program) noexcept {
  if (program.device == nullptr || program.device->backend != Backend::Cpu ||
      program.cpu_graph == nullptr || program.cpu_graph->runtime == nullptr) {
    return 0u;
  }
  std::uint64_t total = 0u;
  for (const CpuRuntimeStep &runtime_step : program.cpu_graph->runtime->steps) {
    const auto *const primitive =
        std::get_if<CpuRuntimePrimitive>(&runtime_step);
    if (primitive == nullptr) {
      continue;
    }
    std::uint64_t count = 0u;
    switch (primitive->kind) {
    case Primitive::Factor:
      if (const auto *const plan =
              std::get_if<kernel::FactorPlan>(&primitive->plan)) {
        count = plan->status_count;
      }
      break;
    case Primitive::Solve:
      if (const auto *const plan =
              std::get_if<kernel::SolvePlan>(&primitive->plan)) {
        count = plan->status_count;
      }
      break;
    case Primitive::Spectrum:
      if (const auto *const plan =
              std::get_if<kernel::SpectrumPlan>(&primitive->plan)) {
        count = plan->status_count;
      }
      break;
    default:
      break;
    }
    total = ::rund::detail::counter::SaturatingAdd(total, count);
  }
  return total;
}

} // namespace rund::compute::detail
