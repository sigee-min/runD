#include "local.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/accel/kernel/prepared/failure.hpp"
#include "src/compute/job/control/model.hpp"
#include "src/compute/job/local.hpp"
#include "src/compute/memory/arena.hpp"
#include "src/compute/memory/cpu.hpp"
#include "src/compute/pipeline/plan/arena.hpp"
#include "src/compute/pipeline/plan/local.hpp"
#include "src/compute/pipeline/state.hpp"
#include "src/compute/status.hpp"

#include <rund/counter.hpp>

#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include <memory>

namespace rund_node_test_pipeline::memory {

[[nodiscard]] int CheckBasic(rund::compute::Device &device) {
  using namespace rund::compute;
  if (detail::project_pipeline_preparation_reason(
          rund::node::accel::detail::
              PreparedPipelineTemplateStepCapacityReasonKey) !=
      Reason::PipelineCapacity) {
    return 21;
  }
  constexpr std::array<std::int32_t, 4u> first_input{1, 2, 3, 4};
  constexpr std::array<std::int32_t, 4u> second_input{2, 3, 4, 5};
  const auto build = [&](const char *name, const std::int32_t add) {
    return on(device)
        .map<std::int32_t>(name, first_input.size(),
                           [](auto value) { return value * 2; })
        .scan(Scan::InclusiveSum)
        .map(name,
             capture([](auto value, auto constant) { return value + constant; },
                     add))
        .scan(Scan::InclusiveSum)
        .compile();
  };
  auto first = build("pipeline memory first", 1);
  auto second = build("pipeline memory second", 2);
  auto first_source = Upload(device, first_input);
  auto second_source = Upload(device, second_input);
  auto first_output = device.buffer<std::int32_t>(first_input.size());
  auto second_output = device.buffer<std::int32_t>(second_input.size());
  if (!first || !second || !first_source || !second_source || !first_output ||
      !second_output) {
    return 1;
  }
  const std::shared_ptr<detail::ProgramState> &first_program =
      detail::ProgramAccess::state(*first);
  auto standalone = first->resident(first_input);
  const std::shared_ptr<detail::JobState> standalone_state =
      standalone ? detail::JobAccess::state(*standalone)
                 : std::shared_ptr<detail::JobState>{};
  auto missing_workspace = std::make_shared<detail::JobState>();
  missing_workspace->program = first_program;
  const Status missing_status = detail::prepare_job_state(
      missing_workspace, detail::JobBindings::ReadOnly,
      detail::JobGraphBufferMode::SealedPipeline);
  auto invalid_mode = std::make_shared<detail::JobState>();
  invalid_mode->program = first_program;
  const Status invalid_mode_status =
      detail::prepare_job_state(invalid_mode, detail::JobBindings::ReadOnly,
                                static_cast<detail::JobGraphBufferMode>(0xffu));
  auto prepopulated = std::make_shared<detail::JobState>();
  prepopulated->program = first_program;
  prepopulated->graph_buffers.resize(1u);
  const Status prepopulated_status =
      detail::prepare_job_state(prepopulated, detail::JobBindings::ReadOnly,
                                detail::JobGraphBufferMode::SealedPipeline);
  if (first_program == nullptr || first_program->chunks.empty() ||
      standalone_state == nullptr || standalone_state->workspace != nullptr ||
      standalone_state->graph_buffers.size() != first_program->chunks.size() ||
      missing_status || missing_status.reason() != Reason::PipelineInvalid ||
      !missing_workspace->graph_buffers.empty() || invalid_mode_status ||
      invalid_mode_status.reason() != Reason::PipelineInvalid ||
      !invalid_mode->graph_buffers.empty() || prepopulated_status ||
      prepopulated_status.reason() != Reason::PipelineInvalid ||
      prepopulated->graph_buffers.size() != 1u ||
      prepopulated->graph_buffers.front() != nullptr) {
    return 27;
  }
  auto builder = pipeline(device)
                     .then(*first, read(*first_source), write(*first_output))
                     .then(*second, read(*second_source), write(*second_output));
  const auto plan = builder.plan();
  if (!plan ||
      plan->peak_bytes !=
          plan->state_bytes + plan->transient_bytes + plan->prepared_bytes ||
      plan->prepared_command_count != 2u || plan->barrier_count != 1u) {
    return 2;
  }
  const bool accelerated =
      detail::DeviceAccess::state(device)->backend != Backend::Cpu;
  const auto device_info = device.info();
  const std::uint64_t expected_scratch =
      accelerated && device_info ? device_info->storage_alignment : 0u;
  const std::uint64_t expected_scratch_payload =
      accelerated ? sizeof(std::int32_t) : 0u;
  if ((accelerated &&
       (!device_info || device_info->storage_alignment == 0u ||
        plan->scratch_payload_bytes != expected_scratch_payload ||
        plan->scratch_bytes != expected_scratch || plan->scratch_count != 1u ||
        plan->scratch_payload_bytes > plan->scratch_bytes ||
        plan->scratch_bytes > plan->prepared_bytes)) ||
      (!accelerated &&
       (plan->scratch_payload_bytes != 0u || plan->scratch_bytes != 0u ||
        plan->scratch_count != 0u))) {
    return 16;
  }
  auto prepared = std::move(builder).prepare();
  const std::shared_ptr<detail::PipelineState> state =
      prepared ? detail::PipelineStateAccess::state(*prepared)
               : std::shared_ptr<detail::PipelineState>{};
  if (state == nullptr || prepared->plan() != *plan ||
      state->steps.size() != 2u ||
      state->steps[0u].program == state->steps[1u].program ||
      state->steps[0u].job == nullptr || state->steps[1u].job == nullptr ||
      state->steps[0u].job->workspace == nullptr ||
      state->steps[1u].job->workspace == nullptr ||
      state->steps[0u].job->workspace == state->steps[1u].job->workspace) {
    return 3;
  }
  if (!accelerated) {
    const std::shared_ptr<detail::CpuPreparedArena> &cpu_arena =
        state->cpu_prepared_arena;
    if (cpu_arena == nullptr || cpu_arena->payload_host_bytes() >
                                    std::numeric_limits<std::uint64_t>::max() -
                                        cpu_arena->payload_tile_bytes()) {
      return 20;
    }
    const std::uint64_t arena_payload =
        cpu_arena->payload_host_bytes() + cpu_arena->payload_tile_bytes();
    const std::uint64_t arena_extent = cpu_arena->extent_bytes();
    const std::uint64_t arena_committed = cpu_arena->committed_bytes();
    if (arena_payload > arena_extent || arena_extent > arena_committed ||
        arena_payload > plan->peak_bytes ||
        arena_committed > std::numeric_limits<std::uint64_t>::max() -
                              (plan->peak_bytes - arena_payload) ||
        plan->arena_extent_bytes != arena_extent ||
        plan->committed_peak_bytes !=
            plan->peak_bytes - arena_payload + arena_committed) {
      return 20;
    }
  } else if (plan->arena_extent_bytes != 0u ||
             plan->committed_peak_bytes != plan->peak_bytes) {
    return 20;
  }
  const MemoryStats prepared_memory = prepared->memory();
  const std::uint64_t planned_resident =
      plan->state_bytes + plan->transient_bytes + plan->prepared_buffer_bytes;
  const std::uint64_t planned_host =
      planned_resident + plan->prepared_host_bytes;
  if ((!accelerated &&
       (plan->prepared_native_bytes != 0u ||
        prepared_memory.resident.current != planned_resident ||
        prepared_memory.host.current > planned_host ||
        prepared_memory.tile.current != plan->prepared_tile_bytes ||
        prepared_memory.host.current + prepared_memory.tile.current >
            plan->peak_bytes)) ||
      prepared_memory.resident.current > plan->peak_bytes ||
      prepared_memory.transfer.current != 0u ||
      prepared_memory.transfer.reused != 0u ||
      prepared_memory.transfer.budget != 0u) {
    std::fprintf(
        stderr,
        "prepared memory backend=%u resident=%llu/%llu host=%llu/%llu "
        "tile=%llu/%llu native=%llu peak=%llu\n",
        static_cast<unsigned>(prepared_memory.backend),
        static_cast<unsigned long long>(prepared_memory.resident.current),
        static_cast<unsigned long long>(planned_resident),
        static_cast<unsigned long long>(prepared_memory.host.current),
        static_cast<unsigned long long>(planned_host),
        static_cast<unsigned long long>(prepared_memory.tile.current),
        static_cast<unsigned long long>(plan->prepared_tile_bytes),
        static_cast<unsigned long long>(plan->prepared_native_bytes),
        static_cast<unsigned long long>(plan->peak_bytes));
    return 19;
  }
  constexpr std::array<std::int32_t, 4u> rewritten_first{1, 2, 3, 4};
  WriteStats pipeline_writes{};
  const MemoryStats before_write = prepared->memory();
  const Status pipeline_write = detail::write_pipeline_raw(
      state, detail::BufferAccess::state(*first_source),
      detail::HostView{rewritten_first.data(), rewritten_first.size(),
                       detail::type<std::int32_t>()},
      pipeline_writes);
  const MemoryStats after_write = prepared->memory();
  constexpr std::uint64_t pipeline_write_bytes =
      rewritten_first.size() * sizeof(std::int32_t);
  if (!pipeline_write || pipeline_writes.bytes != pipeline_write_bytes ||
      after_write.transfer.current != 0u ||
      after_write.transfer.peak !=
          std::max(before_write.transfer.peak, pipeline_write_bytes) ||
      after_write.transfer.cumulative !=
          ::rund::detail::counter::SaturatingAdd(
              before_write.transfer.cumulative, pipeline_write_bytes) ||
      after_write.transfer.reused != 0u || after_write.transfer.budget != 0u ||
      after_write.resident.current != before_write.resident.current) {
    return 29;
  }
  const auto &first_arena = state->steps[0u].job->workspace->arena;
  const auto &second_arena = state->steps[1u].job->workspace->arena;
  if ((accelerated && (first_arena == nullptr || first_arena != second_arena ||
                       first_arena->scratch.empty())) ||
      (!accelerated && first_arena != nullptr &&
       !first_arena->scratch.empty())) {
    return 17;
  }
  std::array<MemoryEntry, 32u> entries{};
  const MemorySnapshot snapshot = prepared->memory_snapshot(entries);
  std::uint64_t resident_scratch = 0u;
  std::uint64_t device_scratch = 0u;
  std::uint64_t physical_scratch = 0u;
  if (plan->scratch_count <= state->prepared_buffers.size()) {
    const std::size_t first_scratch =
        state->prepared_buffers.size() -
        static_cast<std::size_t>(plan->scratch_count);
    for (std::size_t index = first_scratch;
         index < state->prepared_buffers.size(); ++index) {
      physical_scratch += state->prepared_buffers[index]->physical_bytes;
    }
  }
  for (std::size_t index = 0u; index < snapshot.written; ++index) {
    const MemoryEntry &entry = entries[index];
    if (entry.use != MemoryUse::Scratch) {
      continue;
    }
    if (entry.category == MemoryCategory::Resident) {
      resident_scratch += entry.bytes.current;
    } else if (entry.category == MemoryCategory::Device) {
      device_scratch += entry.bytes.current;
    }
  }
  if (snapshot.truncated() ||
      (accelerated && (resident_scratch != plan->scratch_bytes ||
                       physical_scratch < plan->scratch_bytes ||
                       device_scratch != physical_scratch)) ||
      (!accelerated && (resident_scratch != 0u || device_scratch != 0u))) {
    return 18;
  }
  const auto &first_buffers = state->steps[0u].job->workspace->buffers;
  const auto &second_buffers = state->steps[1u].job->workspace->buffers;
  const bool shared = std::any_of(
      first_buffers.begin(), first_buffers.end(), [&](const auto &buffer) {
        return std::find(second_buffers.begin(), second_buffers.end(), buffer) !=
               second_buffers.end();
      });
  if (!shared || prepared->stats().pipeline.barrier_count != 1u ||
      !prepared->run()) {
    return 4;
  }
  std::array<std::int32_t, 4u> first_observed{};
  std::array<std::int32_t, 4u> second_observed{};
  if (!ReadExact(*prepared, *first_output, first_observed) ||
      !ReadExact(*prepared, *second_output, second_observed) ||
      first_observed != std::array<std::int32_t, 4u>{3, 10, 23, 44} ||
      second_observed != std::array<std::int32_t, 4u>{6, 18, 38, 68}) {
    return 5;
  }
  const MemoryStats observed_memory = prepared->memory();
  std::array<MemoryEntry, 32u> observed_entries{};
  const MemorySnapshot observed_snapshot =
      prepared->memory_snapshot(observed_entries);
  std::size_t traffic_rows = 0u;
  bool traffic_exact = true;
  for (std::size_t index = 0u; index < observed_snapshot.written; ++index) {
    const MemoryEntry &entry = observed_entries[index];
    if (entry.category != MemoryCategory::Transfer ||
        entry.use != MemoryUse::Traffic) {
      continue;
    }
    ++traffic_rows;
    traffic_exact =
        traffic_exact && entry.bytes.current == 0u &&
        entry.bytes.peak == observed_memory.transfer.peak &&
        entry.bytes.cumulative == observed_memory.transfer.cumulative &&
        entry.bytes.reused == 0u && entry.bytes.budget == 0u;
  }
  const bool has_traffic = observed_memory.transfer.peak != 0u ||
                           observed_memory.transfer.cumulative != 0u;
  if (observed_snapshot.truncated() || observed_memory.transfer.current != 0u ||
      observed_memory.transfer.reused != 0u ||
      observed_memory.transfer.budget != 0u || !traffic_exact ||
      traffic_rows != (has_traffic ? 1u : 0u)) {
    return 28;
  }
  return 0;
}

} // namespace rund_node_test_pipeline::memory
