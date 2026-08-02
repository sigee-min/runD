#include "model.hpp"

#include "../../cpu/bounded.hpp"
#include "../../cpu/graph.hpp"
#include "../../cpu/map.hpp"
#include "../../cpu/run/state.hpp"
#include "../../cpu/view.hpp"
#include "../../type.hpp"
#include "../state.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <variant>

namespace rund::compute::detail {

Status prepare_graph_step(JobState &job,
                          const std::atomic_bool *const cancel) noexcept {
  CpuRun &run = *job.cpu;
  const CpuGraphProgram &program = *job.program->cpu_graph;
  const CpuRuntimeGraph &graph = *program.runtime;
  if (run.graph == nullptr) {
    return Status::fail(Reason::CpuRunInvalid);
  }
  run.pending_dispatches = 0u;
  const auto buffer = [&](const std::uint32_t value) noexcept {
    return graph_value_buffer_id(*job.program, value, job.inputs, job.outputs,
                                 run.graph->buffers);
  };
  const auto view = [&](const std::uint32_t value) noexcept {
    BufferState *const owner = buffer(value);
    return owner == nullptr ? JobBufferView{}
                            : job_value_view(job, value, *owner);
  };
  const auto type = [&](const std::uint32_t value) noexcept {
    return value == 0u || value > graph.values.size()
               ? Type::I32
               : graph.values[value - 1u].type;
  };
  const auto control =
      [](const CpuRuntimeStep &step) noexcept -> const FlowControl & {
    if (const auto *map = std::get_if<CpuRuntimeMap>(&step)) {
      return map->control;
    }
    if (const auto *scan = std::get_if<CpuRuntimeScan>(&step)) {
      return scan->control;
    }
    return std::get<CpuRuntimePrimitive>(step).control;
  };
  run.controlled_count = 0u;
  run.controlled_count_valid = false;
  while (run.step < graph.steps.size()) {
    const Status reset = reset_cpu(job, run.step);
    if (!reset) {
      return reset;
    }
    const FlowControl active = control(graph.steps[run.step]);
    if (active.empty()) {
      break;
    }
    run.controlled_count = 0u;
    run.controlled_count_valid = false;
    if (active.predicate != 0u) {
      std::uint64_t predicate = 0u;
      BufferState *const owner = buffer(active.predicate);
      const JobBufferView value =
          owner == nullptr ? JobBufferView{} : view(active.predicate);
      const Status observed =
          read_control_value(owner, value, type(active.predicate), predicate);
      if (!observed) {
        return observed;
      }
      if (predicate != active.predicate_expected) {
        ::rund::detail::counter::Accumulate(
            run.stats.control.skipped_iteration_count, 1u);
        ++run.step;
        continue;
      }
      ::rund::detail::counter::Accumulate(run.stats.control.iteration_count,
                                          1u);
    }
    if (active.count != 0u) {
      std::uint64_t logical = 0u;
      BufferState *const owner = buffer(active.count);
      const JobBufferView value =
          owner == nullptr ? JobBufferView{} : view(active.count);
      const Status observed =
          read_control_value(owner, value, type(active.count), logical);
      if (!observed) {
        return observed;
      }
      ::rund::detail::counter::Accumulate(
          run.stats.control.generated_item_count, logical);
      ::rund::detail::counter::Accumulate(run.stats.control.generated_capacity,
                                          active.capacity);
      if (logical > active.capacity ||
          logical > std::numeric_limits<kernel::u32>::max()) {
        run.stats.control.overflow_ordinal =
            std::min(run.stats.control.overflow_ordinal,
                     static_cast<std::uint64_t>(active.capacity));
        return Status::fail(Reason::WorksetOverflow);
      }
      if (active.iteration != 0u && active.predicate == 0u) {
        if (logical == 0u) {
          ::rund::detail::counter::Accumulate(
              run.stats.control.skipped_iteration_count, 1u);
          ++run.step;
          continue;
        }
        ::rund::detail::counter::Accumulate(run.stats.control.iteration_count,
                                            1u);
      }
      ::rund::detail::counter::Accumulate(
          run.stats.control.indirect_dispatch_count, 1u);
      ::rund::detail::counter::Accumulate(
          run.stats.control.indirect_work_item_count, logical);
      run.controlled_count = logical;
      run.controlled_count_valid = true;
    }
    break;
  }
  if (run.step >= graph.steps.size()) {
    return Status::success();
  }
  const CpuRuntimeStep &step = graph.steps[run.step];
  if (std::holds_alternative<CpuRuntimeMap>(step)) {
    CpuMapRun *const stored_map =
        run.graph->storage == nullptr
            ? nullptr
            : cpu_map_run(*run.graph->storage, run.step);
    CpuMapRoute *const stored_route = cpu_map_route(*run.graph, run.step);
    if (run.graph->storage == nullptr || run.step >= program.maps.size() ||
        stored_map == nullptr || stored_route == nullptr ||
        program.maps[run.step] == nullptr) {
      return Status::fail(Reason::GraphBindingInvalid);
    }
    CpuMapRun &map = *stored_map;
    CpuMapRoute &route = *stored_route;
    CpuProgram &kernel = *program.maps[run.step];
    const std::size_t count =
        run.controlled_count_valid
            ? static_cast<std::size_t>(run.controlled_count)
            : kernel.tile_plan.count();
    if (!route.bindings_frozen) {
      return Status::fail(Reason::GraphBindingInvalid);
    }
    const Status bound = begin_cpu_map(kernel, map, route, count,
                                       run.graph->overflow_ordinal, cancel);
    if (!bound) {
      return bound;
    }
    run.pass = CpuPass::Map;
    return Status::success();
  }
  if (const auto *scan = std::get_if<CpuRuntimeScan>(&step)) {
    BufferState *const input = buffer(scan->input);
    BufferState *const output = buffer(scan->output);
    const JobBufferView input_view =
        input == nullptr ? JobBufferView{} : view(scan->input);
    const JobBufferView output_view =
        output == nullptr ? JobBufferView{} : view(scan->output);
    const std::optional<CpuView> source = cpu_view(input, input_view);
    const std::optional<CpuView> target = cpu_view(output, output_view);
    CpuCollectiveRun *const collective =
        run.graph->storage == nullptr
            ? nullptr
            : cpu_collective_run(*run.graph->storage, run.step);
    const Type input_type = graph.values[scan->input - 1u].type;
    const Type output_type = graph.values[scan->output - 1u].type;
    if (!source || !target || source->data == nullptr ||
        target->data == nullptr || collective == nullptr ||
        !source->footprint.dense() || !target->footprint.dense() ||
        input_view.element_bytes != type_bytes(input_type) ||
        output_view.element_bytes != type_bytes(output_type)) {
      return Status::fail(Reason::CpuBufferInvalid);
    }
    kernel::u32 logical = 0u;
    if (scan->count != 0u) {
      const Status count =
          read_bounded_count(buffer(scan->count), view(scan->count),
                             type(scan->count), input_view.count, logical);
      if (!count) {
        return count;
      }
    } else if (input_view.count >
               std::numeric_limits<kernel::u32>::max()) {
      return Status::fail(Reason::TileRunCapacity);
    } else {
      logical = static_cast<kernel::u32>(input_view.count);
    }
    const Status prepared = prepare_bounded_collective(*collective, logical);
    if (!prepared) {
      return prepared;
    }
    run.tile = CpuCollectiveTileContext{
        .run = collective,
        .input = source->data,
        .output = target->data,
        .kind = CpuCollectiveKind::Scan,
        .pass = CpuPass::ScanLocal,
        .scan = scan->operation,
        .type = input_type,
        .cancel = cancel,
    };
    run.pass = CpuPass::ScanLocal;
    return Status::success();
  }
  const auto *primitive = std::get_if<CpuRuntimePrimitive>(&step);
  if (primitive == nullptr) {
    return Status::fail(Reason::GraphStepInvalid);
  }
  if (primitive->kind != Primitive::Reduce) {
    run.pass = CpuPass::Primitive;
    return Status::success();
  }
  BufferState *const input = buffer(primitive->inputs.front());
  BufferState *const output = buffer(primitive->output);
  const JobBufferView input_view =
      input == nullptr ? JobBufferView{} : view(primitive->inputs.front());
  const JobBufferView output_view =
      output == nullptr ? JobBufferView{} : view(primitive->output);
  const std::optional<CpuView> source = cpu_view(input, input_view);
  const std::optional<CpuView> target = cpu_view(output, output_view);
  CpuCollectiveRun *const collective =
      run.graph->storage == nullptr
          ? nullptr
          : cpu_collective_run(*run.graph->storage, run.step);
  const Type input_type = graph.values[primitive->inputs.front() - 1u].type;
  const Type output_type = graph.values[primitive->output - 1u].type;
  if (!source || !target || source->data == nullptr ||
      target->data == nullptr || collective == nullptr ||
      !source->footprint.dense() || !target->footprint.dense() ||
      input_view.element_bytes != type_bytes(input_type) ||
      output_view.element_bytes != type_bytes(output_type)) {
    return Status::fail(Reason::CpuBufferInvalid);
  }
  kernel::u32 logical = 0u;
  if (primitive->inputs.size() == 2u) {
    const Status count = read_bounded_count(
        buffer(primitive->inputs[1u]), view(primitive->inputs[1u]),
        type(primitive->inputs[1u]), input_view.count, logical);
    if (!count) {
      return count;
    }
    const kernel::ReduceOp operation =
        std::get<kernel::ReducePlan>(primitive->plan).op;
    if (logical == 0u && (operation == kernel::ReduceOp::Min ||
                          operation == kernel::ReduceOp::Max)) {
      return Status::fail(Reason::ReduceCountZero);
    }
  } else if (input_view.count > std::numeric_limits<kernel::u32>::max()) {
    return Status::fail(Reason::TileRunCapacity);
  } else {
    logical = static_cast<kernel::u32>(input_view.count);
  }
  const Status prepared = prepare_bounded_collective(*collective, logical);
  if (!prepared) {
    return prepared;
  }
  run.tile = CpuCollectiveTileContext{
      .run = collective,
      .input = source->data,
      .output = target->data,
      .kind = CpuCollectiveKind::Reduce,
      .pass = CpuPass::ReduceLocal,
      .reduce = std::get<kernel::ReducePlan>(primitive->plan).op,
      .type = input_type,
      .cancel = cancel,
  };
  run.pass = CpuPass::ReduceLocal;
  return Status::success();
}

Status initialize_cpu_run(JobState &job) noexcept {
  if (job.program == nullptr || job.program->cpu_graph == nullptr ||
      job.program->cpu_graph->runtime == nullptr || job.cpu == nullptr ||
      job.cpu->graph == nullptr || job.inputs.empty() || job.outputs.empty()) {
    return Status::fail(Reason::CpuRunInvalid);
  }
  if (job.cpu->graph->bound_inputs != job.inputs.data()) {
    const Status rebound = refresh_cpu_map_bindings(job);
    if (!rebound) {
      return rebound;
    }
  }
  job.cpu->step = 0u;
  job.cpu->reset = 0u;
  job.cpu->pass = CpuPass::None;
  job.cpu->pending_dispatches = 0u;
  job.cpu->controlled_count = 0u;
  job.cpu->controlled_count_valid = false;
  job.cpu->graph->semantic_primitive = Primitive::Reduce;
  job.cpu->graph->semantic_status = 0u;
  job.cpu->graph->semantic_failure_count = 0u;
  job.cpu->graph->conflict_count = 0u;
  job.cpu->graph->overflow_ordinal = ControlStats::no_overflow;
  job.cpu->stats = Stats{
      .backend = Backend::Cpu,
      .graph_read_bytes = job.program->graph_info.read_bytes,
      .graph_hash = job.program->cpu_graph->graph_hash,
  };
  return Status::success();
}

namespace {

[[nodiscard]] StepResult
next_cpu(JobState &job, const std::atomic_bool *const cancel) noexcept {
  const Status prepared = prepare_graph_step(job, cancel);
  if (!prepared) {
    return {.status = prepared};
  }
  const CpuRuntimeGraph &graph = *job.program->cpu_graph->runtime;
  return {.complete = job.cpu->step >= graph.steps.size()};
}

} // namespace

StepResult start_cpu(JobState &job,
                     const std::atomic_bool *const cancel) noexcept {
  const Status initialized = initialize_cpu_run(job);
  return initialized ? next_cpu(job, cancel)
                     : StepResult{.status = initialized};
}

StepResult finish_cpu(JobState &job,
                      const kernel::ComputeTileRunResult *const tiles,
                      const std::atomic_bool *const cancel) noexcept {
  const PassResult finished = finish_graph_pass(job, tiles, cancel);
  if (!finished) {
    return {.status = finished.status};
  }
  return finished.flow == PassFlow::Repeat ? StepResult{}
                                           : next_cpu(job, cancel);
}

} // namespace rund::compute::detail
