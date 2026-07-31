#include "state.hpp"

#include "../../buffer/local.hpp"
#include "../graph.hpp"
#include "../map.hpp"
#include "../scratch.hpp"

#include <array>
#include <new>
#include <utility>

namespace rund::compute::detail {

CpuGraphProgram::~CpuGraphProgram() = default;

namespace {

[[nodiscard]] Result<std::unique_ptr<CpuMapRun>>
make_map_run(const CpuProgram &program) {
  try {
    auto run = std::make_unique<CpuMapRun>();
    run->tiles = program.tile_plan.make_run();
    if (!run->tiles.prepared()) {
      return Result<std::unique_ptr<CpuMapRun>>::fail(Reason::TileRunCapacity);
    }
    run->scratch.resize(program.workers);
    run->simd.resize(program.workers);
    for (auto &scratch : run->scratch) {
      scratch.allocate(program.scratch_words);
    }
    return Result<std::unique_ptr<CpuMapRun>>::success(std::move(run));
  } catch (const std::bad_alloc &) {
    return Result<std::unique_ptr<CpuMapRun>>::fail(Reason::TileRunCapacity);
  }
}

[[nodiscard]] Result<std::unique_ptr<CpuCollectiveRun>>
make_collective_run(const CpuCollective &program) {
  try {
    auto run = std::make_unique<CpuCollectiveRun>();
    run->tiles = program.tile_plan.make_run();
    if (!run->tiles.prepared()) {
      return Result<std::unique_ptr<CpuCollectiveRun>>::fail(
          Reason::TileRunCapacity);
    }
    run->totals.resize(program.tile_count);
    if (program.needs_prefixes) {
      run->prefixes.resize(program.tile_count);
    }
    run->tile_size = program.tile_size;
    run->needs_prefixes = program.needs_prefixes;
    return Result<std::unique_ptr<CpuCollectiveRun>>::success(std::move(run));
  } catch (const std::bad_alloc &) {
    return Result<std::unique_ptr<CpuCollectiveRun>>::fail(
        Reason::TileRunCapacity);
  }
}

[[nodiscard]] Status freeze_cpu_map_bindings(JobState &job,
                                             CpuGraphRun &run) noexcept {
  if (job.program == nullptr || job.program->device == nullptr ||
      job.program->cpu_graph == nullptr ||
      job.program->cpu_graph->runtime == nullptr) {
    return Status::fail(Reason::CpuRuntimeInvalid);
  }
  CpuGraphProgram &program = *job.program->cpu_graph;
  const CpuRuntimeGraph &runtime = *program.runtime;
  if (program.maps.size() != runtime.steps.size() ||
      run.maps.size() != runtime.steps.size()) {
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
  for (std::size_t step = 0u; step < runtime.steps.size(); ++step) {
    const auto *const map = std::get_if<CpuRuntimeMap>(&runtime.steps[step]);
    if (map == nullptr) {
      continue;
    }
    CpuProgram *const map_program = program.maps[step].get();
    CpuMapRun *const map_run = run.maps[step].get();
    if (map_program == nullptr || map_run == nullptr ||
        map->inputs.size() > kernel::kMaxComputeBindingCount ||
        map->outputs.empty() ||
        map->outputs.size() > MaxOutputs) {
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
        *map_program, job.program->device, *map_run,
        std::span<BufferState *const>{inputs.data(), map->inputs.size()},
        std::span<BufferState *const>{outputs.data(), map->outputs.size()},
        std::span<const JobBufferView>{input_views.data(), map->inputs.size()},
        std::span<const JobBufferView>{output_views.data(),
                                       map->outputs.size()});
    if (!prepared) {
      return prepared;
    }
    map_run->bindings_frozen = true;
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

Result<std::unique_ptr<CpuRun>>
make_cpu_run(const std::shared_ptr<ProgramState> &program,
             const std::span<const std::shared_ptr<BufferState>> workspace) {
  if (program == nullptr || program->cpu_graph == nullptr) {
    return Result<std::unique_ptr<CpuRun>>::success(nullptr);
  }
  if (program->cpu_graph->runtime == nullptr) {
    return Result<std::unique_ptr<CpuRun>>::fail(Reason::CpuRuntimeInvalid);
  }
  try {
    auto cpu = std::make_unique<CpuRun>();
    const CpuGraphProgram &graph_program = *program->cpu_graph;
    const CpuRuntimeGraph &graph = *graph_program.runtime;
    auto run = std::make_unique<CpuGraphRun>();
    run->maps.resize(graph.steps.size());
    run->collectives.resize(graph.steps.size());
    if (!workspace.empty()) {
      if (workspace.size() != program->chunks.size()) {
        return Result<std::unique_ptr<CpuRun>>::fail(Reason::PipelineInvalid);
      }
      run->buffers = workspace;
    } else {
      run->owned_buffers.reserve(program->chunks.size());
      for (const Chunk chunk : program->chunks) {
        auto buffer = make_workspace_buffer(program->device, chunk.count);
        if (!buffer) {
          return Result<std::unique_ptr<CpuRun>>::fail(buffer.reason());
        }
        run->owned_buffers.push_back(std::move(buffer).value());
      }
      run->buffers = run->owned_buffers;
    }
    for (std::size_t index = 0u; index < graph.steps.size(); ++index) {
      if (graph_program.maps[index] != nullptr) {
        auto map = make_map_run(*graph_program.maps[index]);
        if (!map) {
          return Result<std::unique_ptr<CpuRun>>::fail(map.reason());
        }
        run->maps[index] = std::move(map).value();
      }
      if (graph_program.collectives[index] != nullptr) {
        auto collective =
            make_collective_run(*graph_program.collectives[index]);
        if (!collective) {
          return Result<std::unique_ptr<CpuRun>>::fail(collective.reason());
        }
        run->collectives[index] = std::move(collective).value();
      }
      const auto *primitive =
          std::get_if<CpuRuntimePrimitive>(&graph.steps[index]);
      if (primitive == nullptr || primitive->kind == Primitive::Reduce) {
        continue;
      }
      auto scratch = prepare_cpu_scratch(*primitive);
      if (!scratch) {
        return Result<std::unique_ptr<CpuRun>>::fail(scratch.reason());
      }
      if (std::holds_alternative<std::monostate>(scratch.value())) {
        continue;
      }
      if (run->scratch.empty()) {
        run->scratch.resize(graph.steps.size());
      }
      run->scratch[index] = std::move(scratch).value();
    }
    cpu->graph = std::move(run);
    return Result<std::unique_ptr<CpuRun>>::success(std::move(cpu));
  } catch (const std::bad_alloc &) {
    return Result<std::unique_ptr<CpuRun>>::fail(Reason::BufferCapacity);
  }
}

Status prepare_cpu_run(JobState &job) {
  if (job.program != nullptr && job.program->empty()) {
    return Status::success();
  }
  auto run = make_cpu_run(job.program, job_graph_buffers(job));
  if (!run) {
    return Status::fail(run.reason());
  }
  job.cpu = std::move(run).value();
  return refresh_cpu_map_bindings(job);
}

} // namespace rund::compute::detail
