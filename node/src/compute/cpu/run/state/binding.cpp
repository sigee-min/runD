#include "../../state/program.hpp"
#include "local.hpp"

#include "../../../device/state.hpp"
#include "../../graph.hpp"
#include "../../map.hpp"

#include <array>

namespace rund::compute::detail {
namespace {

[[nodiscard]] Status freeze_cpu_map_bindings(JobState &job,
                                             CpuGraphRun &run) noexcept {
  if (job.program == nullptr || job.program->device == nullptr ||
      job.program->cpu_graph == nullptr ||
      job.program->cpu_graph->runtime == nullptr || run.storage == nullptr ||
      run.storage->program != job.program->cpu_graph.get()) {
    return Status::fail(Reason::CpuRuntimeInvalid);
  }
  CpuGraphProgram &program = *job.program->cpu_graph;
  const CpuRuntimeGraph &runtime = *program.runtime;
  if (program.maps.size() != runtime.steps.size() ||
      run.storage->map_by_step.size() != runtime.steps.size() ||
      run.maps.size() != run.storage->maps.size()) {
    return Status::fail(Reason::CpuRuntimeInvalid);
  }
  const auto buffer = [&](const std::uint32_t value) noexcept {
    return graph_value_buffer_id(*job.program, value, job.inputs, job.outputs,
                                 run.buffers);
  };
  const auto view = [&](const std::uint32_t value,
                        BufferState *const owner) noexcept {
    return owner == nullptr ? JobBufferView{}
                            : job_value_view(job, value, *owner);
  };
  std::size_t read_begin = 0u;
  std::size_t write_begin = 0u;
  for (std::size_t step = 0u; step < runtime.steps.size(); ++step) {
    const auto *const map = std::get_if<CpuRuntimeMap>(&runtime.steps[step]);
    if (map == nullptr) {
      continue;
    }
    CpuProgram *const map_program = program.maps[step].get();
    CpuMapRun *const map_run = cpu_map_run(*run.storage, step);
    CpuMapRoute *const map_route = cpu_map_route(run, step);
    if (map_program == nullptr || map_run == nullptr || map_route == nullptr ||
        map->inputs.size() > kernel::kMaxComputeBindingCount ||
        map->outputs.empty() || map->outputs.size() > MaxOutputs ||
        read_begin > run.reads.size() ||
        map->inputs.size() > run.reads.size() - read_begin ||
        write_begin > run.writes.size() ||
        map->outputs.size() > run.writes.size() - write_begin) {
      return Status::fail(Reason::GraphBindingInvalid);
    }
    std::array<BufferState *, kernel::kMaxComputeBindingCount> inputs{};
    std::array<BufferState *, MaxOutputs> outputs{};
    std::array<JobBufferView, kernel::kMaxComputeBindingCount> input_views{};
    std::array<JobBufferView, MaxOutputs> output_views{};
    for (std::size_t index = 0u; index < map->inputs.size(); ++index) {
      inputs[index] = buffer(map->inputs[index]);
      input_views[index] = view(map->inputs[index], inputs[index]);
    }
    for (std::size_t index = 0u; index < map->outputs.size(); ++index) {
      outputs[index] = buffer(map->outputs[index]);
      output_views[index] = view(map->outputs[index], outputs[index]);
    }
    const Status prepared = prepare_cpu_map_bindings(
        *map_program, job.program->device, *map_run, *map_route,
        run.reads.subspan(read_begin, map->inputs.size()),
        run.writes.subspan(write_begin, map->outputs.size()),
        std::span<BufferState *const>{inputs.data(), map->inputs.size()},
        std::span<BufferState *const>{outputs.data(), map->outputs.size()},
        std::span<const JobBufferView>{input_views.data(), map->inputs.size()},
        std::span<const JobBufferView>{output_views.data(),
                                       map->outputs.size()});
    if (!prepared) {
      return prepared;
    }
    map_route->bindings_frozen = true;
    read_begin += map->inputs.size();
    write_begin += map->outputs.size();
  }
  if (read_begin != run.reads.size() || write_begin != run.writes.size()) {
    return Status::fail(Reason::CpuRuntimeInvalid);
  }
  run.bound_inputs = job.inputs.data();
  return Status::success();
}

} // namespace

Status refresh_cpu_map_bindings(JobState &job) noexcept {
  return job.cpu == nullptr || job.cpu->graph == nullptr
             ? Status::success()
             : freeze_cpu_map_bindings(job, *job.cpu->graph);
}

} // namespace rund::compute::detail
