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
  state.stats.control.overflow_ordinal = descriptor->maximum;
  return status;
}

[[nodiscard]] Status read_u32(const std::shared_ptr<BufferState> &owner,
                              const std::size_t offset,
                              std::uint32_t &value) noexcept {
  if (owner == nullptr || owner->type != Type::U32 || offset >= owner->count) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::optional<CpuView> source =
      cpu_view(owner.get(), offset, 1u, 1u, sizeof(std::uint32_t));
  if (!source || source->data == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::memcpy(&value, source->data, sizeof(value));
  return Status::success();
}

[[nodiscard]] Status resident_view(PipelineState &state,
                                   const PipelineWindow &window,
                                   const std::uint32_t output,
                                   const std::uint32_t bank,
                                   CpuView &view) noexcept {
  if (bank > PipelineWindow::second ||
      window.first_step >= state.steps.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::size_t step_index = window.first_step;
  const bool input = bank == PipelineWindow::seed;
  if (bank == PipelineWindow::second && step_index + 1u < state.steps.size() &&
      state.steps[step_index + 1u].window == state.steps[step_index].window) {
    ++step_index;
  }
  const PipelineStep &step = state.steps[step_index];
  const std::shared_ptr<JobState> &job =
      state.transactional && state.attempt_parity != 0u ? step.alternate_job
                                                        : step.job;
  if (job == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::vector<std::shared_ptr<BufferState>> &owners =
      input ? job->inputs : job->outputs;
  const std::vector<JobBufferView> &views =
      input ? job->input_views : job->output_views;
  if (output >= owners.size() || output >= views.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::optional<CpuView> resolved =
      cpu_view(owners[output].get(), views[output]);
  if (!resolved || resolved->data == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  view = *resolved;
  return Status::success();
}

[[nodiscard]] Status seal_resident(PipelineState &state,
                                   const PipelineStep &step,
                                   const PipelineWindow &window) noexcept {
  if (step.iteration_bound == 0u || window.first_step >= state.steps.size()) {
    return Status::success();
  }
  const std::uint32_t final =
      PipelineWindow::first + ((step.iteration_bound - 1u) & 1u);
  if (window.current == final) {
    return Status::success();
  }
  const PipelineStep &first = state.steps[window.first_step];
  const std::shared_ptr<JobState> &job =
      state.transactional && state.attempt_parity != 0u ? first.alternate_job
                                                        : first.job;
  if (job == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  for (std::uint32_t output = 0u; output < job->outputs.size(); ++output) {
    CpuView source{};
    CpuView target{};
    const Status source_ready =
        resident_view(state, window, output, window.current, source);
    const Status target_ready =
        resident_view(state, window, output, final, target);
    if (!source_ready || !target_ready ||
        source.footprint.bytes != target.footprint.bytes ||
        source.footprint.stride != target.footprint.stride ||
        source.footprint.width != target.footprint.width ||
        source.data == nullptr || target.data == nullptr) {
      return Status::fail(Reason::PipelineInvalid);
    }
    if (source.footprint.dense()) {
      std::memmove(target.data, source.data, source.footprint.bytes);
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
  }
  return Status::success();
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

Status cpu_resident_view(PipelineState &state, const PipelineWindow &window,
                         const std::uint32_t output, CpuView &view) noexcept {
  return resident_view(state, window, output, window.current, view);
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
  const Status count_ready =
      read_u32(descriptor->count, descriptor->count_offset, item_count);
  if (!count_ready) {
    return count_ready;
  }
  if (item_count > descriptor->maximum) {
    state.stats.control.overflow_ordinal = descriptor->maximum;
    return Status::fail(Reason::BoundedCountInvalid);
  }
  const std::uint64_t base =
      static_cast<std::uint64_t>(step.iteration) * descriptor->tile;
  bool terminal = false;
  if (descriptor->terminal != std::numeric_limits<std::uint32_t>::max()) {
    CpuView view{};
    const Status terminal_ready = cpu_resident_view(
        state, *descriptor, descriptor->terminal_output, view);
    if (!terminal_ready || view.footprint.count != 1u ||
        view.footprint.width != sizeof(std::uint32_t)) {
      return Status::fail(Reason::PipelineInvalid);
    }
    std::uint32_t observed{};
    std::memcpy(&observed, view.data, sizeof(observed));
    terminal = observed == descriptor->expected;
  }
  if (!descriptor->stopped && base < item_count && !terminal) {
    return Status::success();
  }

  active = false;
  if (!descriptor->stopped) {
    const Status sealed = seal_resident(state, step, *descriptor);
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
