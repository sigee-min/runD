#include "local.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/cpu/state.hpp"
#include "src/compute/job/local.hpp"
#include "src/compute/pipeline/state.hpp"

#include <memory>
#include <vector>

namespace rund_node_test_pipeline {

[[nodiscard]] int CheckSealedPipelineBindings(rund::compute::Device &device) {
  using namespace rund::compute;
  using namespace rund::compute::detail;

  constexpr std::array<std::int32_t, 8u> initial{1, 2, 3, 4, -1, -1, -1, -1};
  constexpr std::array<std::int32_t, 8u> expected{1, 2, 3, 4, 2, 4, 6, 8};
  auto program = on(device)
                     .map<std::int32_t>("pipeline-sealed-binding", 4u,
                                        [](auto value) { return value * 2; })
                     .compile();
  auto storage = Upload(device, initial);
  if (!program || !storage) {
    return 1;
  }
  auto source = storage->view(0u, 4u);
  auto target = storage->view(4u, 4u);
  if (!source || !target) {
    return 2;
  }

  const std::array<ResourceView, 1u> inputs{
      BufferAccess::view(*source, ResourceAccess::Read)};
  const std::array<ResourceView, 1u> outputs{
      BufferAccess::view(*target, ResourceAccess::Write)};
  auto build = make_pipeline(DeviceAccess::state(device));
  append_pipeline(build, ProgramAccess::state(*program), inputs, outputs);
  const auto planned = plan_pipeline(build);
  if (!planned || build == nullptr || build->memory == nullptr ||
      build->steps.size() != 1u || build->memory->step_resources.size() != 1u ||
      build->memory->step_resources[0u].outputs.size() != 1u ||
      build->memory->step_resources[0u].outputs[0u].view.offset != 4u) {
    return 3;
  }

  // The schedule was proved with disjoint input/output Views. Mutate only the
  // stale authored declaration after plan() without invalidating the cached
  // memory plan. Preparation must bind the sealed output at offset 4, not the
  // now-overlapping authored offset 0, and it must not turn the drift into a
  // rejection path.
  build->steps[0u].outputs[0u].offset = 0u;
  auto prepared = prepare_pipeline(build);
  if (!prepared || (*prepared)->steps.size() != 1u ||
      (*prepared)->steps[0u].job == nullptr ||
      (*prepared)->steps[0u].job->output_views.size() != 1u ||
      (*prepared)->steps[0u].job->output_views[0u].offset != 4u) {
    return 4;
  }
  std::array<std::int32_t, 8u> actual{};
  return run_pipeline(*prepared) &&
                 read_pipeline_raw(*prepared, BufferAccess::state(*storage),
                                   Type::I32, FixedFormat{}, actual.data(),
                                   sizeof(actual), actual.size()) &&
                 actual == expected
             ? 0
             : 5;
}

[[nodiscard]] int CheckFrozenCpuMapBindings(rund::compute::Device &device) {
  using namespace rund::compute;
  using namespace rund::compute::detail;
  constexpr std::array<std::int32_t, 4u> input{1, 2, 3, 4};
  auto program = on(device)
                     .map<std::int32_t>("pipeline-frozen-cpu-map", input.size(),
                                        [](auto value) { return value * 2; })
                     .compile();
  auto source = Upload(device, input);
  auto output = device.buffer<std::int32_t>(input.size());
  if (!program || !source || !output) {
    return 1;
  }
  auto prepared_pipeline =
      pipeline(device).then(*program, read(*source), write(*output)).prepare();
  const std::shared_ptr<PipelineState> pipeline_state =
      prepared_pipeline ? PipelineStateAccess::state(*prepared_pipeline)
                        : std::shared_ptr<PipelineState>{};
  const std::shared_ptr<JobState> prepared =
      pipeline_state != nullptr && pipeline_state->steps.size() == 1u
          ? pipeline_state->steps.front().job
          : std::shared_ptr<JobState>{};
  if (prepared == nullptr || prepared->cpu == nullptr ||
      prepared->cpu->graph == nullptr ||
      prepared->cpu->graph->storage == nullptr ||
      prepared->cpu->graph->maps.size() != 1u ||
      prepared->cpu->graph->storage->maps.size() != 1u ||
      cpu_map_run(*prepared->cpu->graph->storage, 0u) == nullptr) {
    return 2;
  }
  CpuGraphRun &graph = *prepared->cpu->graph;
  CpuMapRoute &route = graph.maps.front();
  const auto *const read_data = graph.reads.front().data;
  auto *const write_data = graph.writes.front().data;
  const std::size_t read_stride = graph.reads.front().stride;
  const std::size_t write_stride = graph.writes.front().stride;
  if (!route.bindings_frozen || route.bindings.reads != graph.reads.data() ||
      route.bindings.writes != graph.writes.data() || read_data == nullptr ||
      write_data == nullptr || read_stride != sizeof(std::int32_t) ||
      write_stride != sizeof(std::int32_t)) {
    return 3;
  }
  if (!run_pipeline_job(prepared) || !run_pipeline_job(prepared) ||
      !route.bindings_frozen || route.bindings.reads != graph.reads.data() ||
      route.bindings.writes != graph.writes.data() ||
      graph.reads.front().data != read_data ||
      graph.writes.front().data != write_data ||
      graph.reads.front().stride != read_stride ||
      graph.writes.front().stride != write_stride) {
    return 4;
  }
  auto resident = program->resident(input);
  if (!resident) {
    return 5;
  }
  const std::shared_ptr<JobState> resident_state = JobAccess::state(*resident);
  if (resident_state == nullptr || resident_state->cpu == nullptr ||
      resident_state->cpu->graph == nullptr ||
      resident_state->cpu->graph->storage == nullptr ||
      resident_state->cpu->graph->maps.size() != 1u ||
      resident_state->cpu->graph->storage->maps.size() != 1u ||
      cpu_map_run(*resident_state->cpu->graph->storage, 0u) == nullptr) {
    return 6;
  }
  CpuGraphRun &resident_graph = *resident_state->cpu->graph;
  CpuMapRoute &resident_route = resident_graph.maps.front();
  const auto *const first_input = resident_graph.reads.front().data;
  constexpr std::array<std::int32_t, 4u> next{5, 6, 7, 8};
  if (!resident_route.bindings_frozen || !resident->run() ||
      !resident->write(next) ||
      resident_graph.bound_inputs == resident_state->inputs.data() ||
      !resident->run() ||
      resident_graph.bound_inputs != resident_state->inputs.data() ||
      resident_graph.reads.front().data == first_input) {
    return 7;
  }
  const auto observed = resident->read();
  if (!observed || *observed != std::vector<std::int32_t>{10, 12, 14, 16}) {
    return 8;
  }
  return 0;
}

} // namespace rund_node_test_pipeline
