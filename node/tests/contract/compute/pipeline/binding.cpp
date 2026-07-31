#include "local.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/cpu/state.hpp"
#include "src/compute/job/local.hpp"

#include <memory>
#include <utility>
#include <vector>

namespace rund_node_test_pipeline {

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
  std::vector<std::shared_ptr<BufferState>> inputs{
      BufferAccess::state(*source)};
  std::vector<std::shared_ptr<BufferState>> outputs{
      BufferAccess::state(*output)};
  auto prepared = prepare_pipeline_job_buffers(
      ProgramAccess::state(*program), std::move(inputs), std::move(outputs));
  if (!prepared || prepared.value() == nullptr ||
      prepared.value()->cpu == nullptr ||
      prepared.value()->cpu->graph == nullptr ||
      prepared.value()->cpu->graph->maps.size() != 1u ||
      prepared.value()->cpu->graph->maps.front() == nullptr) {
    return 2;
  }
  CpuMapRun &map = *prepared.value()->cpu->graph->maps.front();
  const auto *const read_data = map.reads.front().data;
  auto *const write_data = map.writes.front().data;
  const std::size_t read_stride = map.reads.front().stride;
  const std::size_t write_stride = map.writes.front().stride;
  if (!map.bindings_frozen || map.bindings.reads != map.reads.data() ||
      map.bindings.writes != map.writes.data() || read_data == nullptr ||
      write_data == nullptr || read_stride != sizeof(std::int32_t) ||
      write_stride != sizeof(std::int32_t)) {
    return 3;
  }
  if (!run_pipeline_job(prepared.value()) ||
      !run_pipeline_job(prepared.value()) || !map.bindings_frozen ||
      map.bindings.reads != map.reads.data() ||
      map.bindings.writes != map.writes.data() ||
      map.reads.front().data != read_data ||
      map.writes.front().data != write_data ||
      map.reads.front().stride != read_stride ||
      map.writes.front().stride != write_stride) {
    return 4;
  }
  auto resident = program->resident(input);
  if (!resident) {
    return 5;
  }
  const std::shared_ptr<JobState> resident_state = JobAccess::state(*resident);
  if (resident_state == nullptr || resident_state->cpu == nullptr ||
      resident_state->cpu->graph == nullptr ||
      resident_state->cpu->graph->maps.size() != 1u ||
      resident_state->cpu->graph->maps.front() == nullptr) {
    return 6;
  }
  CpuGraphRun &resident_graph = *resident_state->cpu->graph;
  CpuMapRun &resident_map = *resident_graph.maps.front();
  const auto *const first_input = resident_map.reads.front().data;
  constexpr std::array<std::int32_t, 4u> next{5, 6, 7, 8};
  if (!resident_map.bindings_frozen || !resident->run() ||
      !resident->write(next) ||
      resident_graph.bound_inputs == resident_state->inputs.data() ||
      !resident->run() ||
      resident_graph.bound_inputs != resident_state->inputs.data() ||
      resident_map.reads.front().data == first_input) {
    return 7;
  }
  const auto observed = resident->read();
  if (!observed || *observed != std::vector<std::int32_t>{10, 12, 14, 16}) {
    return 8;
  }
  return 0;
}

} // namespace rund_node_test_pipeline
