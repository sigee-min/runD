#include "local.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/job/local.hpp"
#include "src/compute/memory/arena.hpp"
#include "src/compute/pipeline/plan/local.hpp"
#include "src/compute/pipeline/state.hpp"

#include <algorithm>
#include <limits>
#include <memory>

namespace rund_node_test_pipeline {

[[nodiscard]] int CheckMemory(rund::compute::Device &device) {
  using namespace rund::compute;
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
  auto builder =
      pipeline(device)
          .then(*first, read(*first_source), write(*first_output))
          .then(*second, read(*second_source), write(*second_output));
  const auto plan = builder.plan();
  if (!plan || plan->peak_bytes != plan->state_bytes + plan->transient_bytes +
                                       plan->prepared_bytes) {
    return 2;
  }
  const bool accelerated =
      detail::DeviceAccess::state(device)->backend != Backend::Cpu;
  const auto device_info = device.info();
  const std::uint64_t expected_scratch =
      accelerated && device_info ? device_info->storage_alignment : 0u;
  if ((accelerated &&
       (!device_info || device_info->storage_alignment == 0u ||
        plan->scratch_bytes != expected_scratch || plan->scratch_count != 1u ||
        plan->scratch_bytes > plan->prepared_bytes)) ||
      (!accelerated &&
       (plan->scratch_bytes != 0u || plan->scratch_count != 0u))) {
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
      state->steps[0u].job->workspace == state->steps[1u].job->workspace ||
      state->shared_buffers.size() != 1u) {
    return 3;
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
        return std::find(second_buffers.begin(), second_buffers.end(),
                         buffer) != second_buffers.end();
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

  // Scatter is a partial writer, so its internal exact-capacity result owns an
  // invocation reset. Distinct Programs must still map that reset chunk into
  // the one cross-Program rank envelope rather than retaining two cold owners.
  const auto partial = [&](const char *name) {
    return on(device)
        .map<std::int32_t>(name, 2u, [](auto value) { return value; })
        .scatter(2u, {.count = 4u})
        .map(name, [](auto value) { return value + 1; })
        .compile();
  };
  auto partial_first = partial("pipeline reset memory first");
  auto partial_second = partial("pipeline reset memory second");
  constexpr std::array<std::int32_t, 2u> first_values{1, 2};
  constexpr std::array<std::int32_t, 2u> second_values{3, 4};
  constexpr std::array<std::uint32_t, 2u> first_indices{0u, 2u};
  constexpr std::array<std::uint32_t, 2u> second_indices{1u, 3u};
  auto first_value_buffer = Upload(device, first_values);
  auto second_value_buffer = Upload(device, second_values);
  auto first_index_buffer = Upload(device, first_indices);
  auto second_index_buffer = Upload(device, second_indices);
  auto partial_first_output = device.buffer<std::int32_t>(4u);
  auto partial_second_output = device.buffer<std::int32_t>(4u);
  if (!partial_first || !partial_second || !first_value_buffer ||
      !second_value_buffer || !first_index_buffer || !second_index_buffer ||
      !partial_first_output || !partial_second_output) {
    return 6;
  }
  auto reset_shared =
      pipeline(device)
          .profile(PipelineProfile::Steps)
          .then(*partial_first, read(*first_value_buffer, *first_index_buffer),
                write(*partial_first_output))
          .then(*partial_second,
                read(*second_value_buffer, *second_index_buffer),
                write(*partial_second_output))
          .prepare();
  const std::shared_ptr<detail::PipelineState> reset_state =
      reset_shared ? detail::PipelineStateAccess::state(*reset_shared)
                   : std::shared_ptr<detail::PipelineState>{};
  const auto reset_chunk = [](const detail::ProgramState &program) {
    for (std::size_t index = 0u; index < program.graph_info.resources.size();
         ++index) {
      if (!program.graph_info.resources[index].requires_reset() ||
          index >= program.graph_value_routes.size()) {
        continue;
      }
      const detail::GraphValueRoute route = program.graph_value_routes[index];
      if (route.source == detail::GraphBindSource::Internal) {
        return static_cast<std::size_t>(route.index);
      }
    }
    return std::numeric_limits<std::size_t>::max();
  };
  if (reset_state == nullptr || reset_state->steps.size() != 2u ||
      reset_state->steps[0u].job == nullptr ||
      reset_state->steps[1u].job == nullptr ||
      reset_state->steps[0u].job->workspace == nullptr ||
      reset_state->steps[1u].job->workspace == nullptr) {
    return 7;
  }
  const std::size_t first_reset = reset_chunk(*reset_state->steps[0u].program);
  const std::size_t second_reset = reset_chunk(*reset_state->steps[1u].program);
  const auto &first_workspace = reset_state->steps[0u].job->workspace->buffers;
  const auto &second_workspace = reset_state->steps[1u].job->workspace->buffers;
  if (first_reset >= first_workspace.size() ||
      second_reset >= second_workspace.size() ||
      first_workspace[first_reset] != second_workspace[second_reset] ||
      reset_state->shared_buffers.size() != 1u ||
      reset_shared->stats().pipeline.barrier_count != 1u ||
      !reset_shared->run()) {
    return 8;
  }
  std::array<std::int32_t, 4u> partial_first_observed{};
  std::array<std::int32_t, 4u> partial_second_observed{};
  const auto exact = [&] {
    return ReadExact(*reset_shared, *partial_first_output,
                     partial_first_observed) &&
           ReadExact(*reset_shared, *partial_second_output,
                     partial_second_observed) &&
           partial_first_observed == std::array<std::int32_t, 4u>{2, 1, 3, 1} &&
           partial_second_observed == std::array<std::int32_t, 4u>{1, 4, 1, 5};
  };
  if (!exact() || !reset_shared->run() || !exact() ||
      reset_shared->generation() != 2u) {
    return 9;
  }
  std::array<PipelineStepProfile, 2u> reset_rows{};
  const auto reset_profile = reset_shared->profile(reset_rows);
  if (!reset_profile || reset_profile->written != reset_rows.size() ||
      !ProfileMemoryReconciles(*reset_profile, reset_rows)) {
    return 10;
  }

  auto arena_program = std::make_shared<detail::ProgramState>();
  arena_program->device = detail::DeviceAccess::state(device);
  arena_program->chunks = {
      detail::Chunk{.count = 9u},
      detail::Chunk{.count = 5u},
  };
  arena_program->chunk_order = {0u, 1u};
  detail::PipelineBuildState arena_build{};
  arena_build.device = detail::DeviceAccess::state(device);
  detail::PipelineBuildStep arena_step{};
  arena_step.program = arena_program;
  arena_step.logical_step = 0u;
  arena_build.steps.push_back(std::move(arena_step));
  const auto arena_plan = detail::plan_memory(arena_build);
  if (!arena_plan || (*arena_plan)->summary.allocation_count != 1u ||
      (*arena_plan)->summary.transient_bytes != 69u * sizeof(std::uint32_t) ||
      (*arena_plan)->steps != std::vector<std::size_t>{0u, 2u} ||
      (*arena_plan)->owners != std::vector<std::size_t>{0u, 0u} ||
      (*arena_plan)->offsets != std::vector<std::size_t>{0u, 64u} ||
      (*arena_plan)->chunks != std::vector<std::size_t>{69u}) {
    return 11;
  }
  const auto arena_memory = detail::make_pipeline_memory(
      arena_build.device, arena_build.steps, **arena_plan);
  if (!arena_memory || arena_memory->buffers.size() != 1u ||
      arena_memory->steps.size() != 1u ||
      arena_memory->steps.front() == nullptr ||
      arena_memory->steps.front()->buffers.size() != 2u ||
      arena_memory->steps.front()->offsets !=
          std::vector<std::size_t>{0u, 64u} ||
      arena_memory->steps.front()->buffers[0u] !=
          arena_memory->buffers.front() ||
      arena_memory->steps.front()->buffers[1u] !=
          arena_memory->buffers.front()) {
    return 12;
  }

  auto chunked = std::make_shared<detail::ProgramState>();
  chunked->device = arena_program->device;
  chunked->chunks = {
      detail::Chunk{
          .count = static_cast<std::size_t>(detail::memory::ChunkWords -
                                            detail::memory::AlignmentWords)},
      detail::Chunk{.count = static_cast<std::size_t>(
                        2u * detail::memory::AlignmentWords)},
  };
  chunked->chunk_order = {0u, 1u};
  detail::PipelineBuildState chunked_build{};
  chunked_build.device = arena_build.device;
  detail::PipelineBuildStep chunked_step{};
  chunked_step.program = chunked;
  chunked_step.logical_step = 0u;
  chunked_build.steps.push_back(std::move(chunked_step));
  const auto chunked_plan = detail::plan_memory(chunked_build);
  const std::size_t chunked_base = static_cast<std::size_t>(
      detail::memory::ChunkWords - detail::memory::AlignmentWords);
  if (!chunked_plan || (*chunked_plan)->summary.allocation_count != 2u ||
      (*chunked_plan)->summary.transient_bytes !=
          static_cast<std::uint64_t>(detail::memory::ChunkWords +
                                     detail::memory::AlignmentWords) *
              sizeof(std::uint32_t) ||
      (*chunked_plan)->summary.reuse_count != 0u ||
      (*chunked_plan)->steps != std::vector<std::size_t>{0u, 2u} ||
      (*chunked_plan)->owners != std::vector<std::size_t>{0u, 1u} ||
      (*chunked_plan)->offsets != std::vector<std::size_t>{0u, 0u} ||
      (*chunked_plan)->chunks !=
          std::vector<std::size_t>{chunked_base,
                                   2u * detail::memory::AlignmentWords}) {
    return 13;
  }

  auto wide = std::make_shared<detail::ProgramState>();
  wide->device = arena_program->device;
  wide->chunks = {
      detail::Chunk{.count = 100u},
      detail::Chunk{.count = 100u},
  };
  wide->chunk_order = {0u, 1u};
  auto tall = std::make_shared<detail::ProgramState>();
  tall->device = arena_program->device;
  tall->chunks = {
      detail::Chunk{.count = 150u},
      detail::Chunk{.count = 1u},
  };
  tall->chunk_order = {0u, 1u};
  detail::PipelineBuildState packed{};
  packed.device = arena_build.device;
  detail::PipelineBuildStep wide_step{};
  wide_step.program = wide;
  wide_step.logical_step = 0u;
  packed.steps.push_back(std::move(wide_step));
  detail::PipelineBuildStep tall_step{};
  tall_step.program = tall;
  tall_step.logical_step = 1u;
  packed.steps.push_back(std::move(tall_step));
  const auto packed_plan = detail::plan_memory(packed);
  if (!packed_plan ||
      (*packed_plan)->summary.transient_bytes != 228u * sizeof(std::uint32_t) ||
      (*packed_plan)->summary.reuse_count != 3u ||
      (*packed_plan)->steps != std::vector<std::size_t>{0u, 2u, 4u} ||
      (*packed_plan)->owners != std::vector<std::size_t>{0u, 0u, 0u, 0u} ||
      (*packed_plan)->offsets != std::vector<std::size_t>{0u, 128u, 0u, 192u} ||
      (*packed_plan)->chunks != std::vector<std::size_t>{228u}) {
    return 14;
  }

  constexpr std::size_t large_words = static_cast<std::size_t>(
      detail::memory::ChunkWords + detail::memory::AlignmentWords);
  constexpr std::size_t larger_words =
      large_words + static_cast<std::size_t>(detail::memory::AlignmentWords);
  auto large = std::make_shared<detail::ProgramState>();
  large->device = arena_program->device;
  large->chunks = {
      detail::Chunk{.count = large_words},
      detail::Chunk{.count = 128u},
  };
  large->chunk_order = {0u, 1u};
  auto larger = std::make_shared<detail::ProgramState>();
  larger->device = arena_program->device;
  larger->chunks = {
      detail::Chunk{.count = larger_words},
      detail::Chunk{.count = 192u},
  };
  larger->chunk_order = {0u, 1u};
  detail::PipelineBuildState dedicated{};
  dedicated.device = arena_build.device;
  detail::PipelineBuildStep large_step{};
  large_step.program = large;
  large_step.logical_step = 3u;
  large_step.iteration = 2u;
  dedicated.steps.push_back(std::move(large_step));
  detail::PipelineBuildStep larger_step{};
  larger_step.program = larger;
  larger_step.logical_step = 4u;
  larger_step.iteration = 7u;
  dedicated.steps.push_back(std::move(larger_step));
  const auto dedicated_plan = detail::plan_memory(dedicated);
  const std::uint64_t larger_bytes =
      static_cast<std::uint64_t>(larger_words) * sizeof(std::uint32_t);
  if (!dedicated_plan ||
      (*dedicated_plan)->summary.transient_bytes !=
          static_cast<std::uint64_t>(larger_words + 192u) *
              sizeof(std::uint32_t) ||
      (*dedicated_plan)->summary.allocation_count != 2u ||
      (*dedicated_plan)->summary.reuse_count != 2u ||
      (*dedicated_plan)->summary.largest_bytes != larger_bytes ||
      (*dedicated_plan)->summary.largest_step != 4u ||
      (*dedicated_plan)->summary.largest_iteration != 7u ||
      (*dedicated_plan)->summary.largest_chunk != 0u ||
      (*dedicated_plan)->steps != std::vector<std::size_t>{0u, 2u, 4u} ||
      (*dedicated_plan)->owners != std::vector<std::size_t>{0u, 1u, 0u, 1u} ||
      (*dedicated_plan)->offsets != std::vector<std::size_t>{0u, 0u, 0u, 0u} ||
      (*dedicated_plan)->chunks !=
          std::vector<std::size_t>{larger_words, 192u}) {
    return 15;
  }

  constexpr std::size_t scale_owners = 8u;
  const std::uint64_t storage_words =
      detail::memory::arena_bytes(*arena_build.device) / detail::memory::Word;
  const std::size_t scale_words = static_cast<std::size_t>(
      std::min(storage_words, detail::memory::ChunkWords));
  auto scale = std::make_shared<detail::ProgramState>();
  scale->device = arena_program->device;
  scale->chunks.assign(scale_owners, detail::Chunk{.count = scale_words});
  for (std::size_t index = 0u; index < scale_owners; ++index) {
    scale->chunk_order.push_back(static_cast<std::uint32_t>(index));
  }
  detail::PipelineBuildState scale_build{};
  scale_build.device = arena_build.device;
  detail::PipelineBuildStep scale_step{};
  scale_step.program = scale;
  scale_step.logical_step = 5u;
  scale_build.steps.push_back(std::move(scale_step));
  const auto scale_plan = detail::plan_memory(scale_build);
  const std::uint64_t scale_bytes =
      static_cast<std::uint64_t>(scale_words) * detail::memory::Word;
  if (!scale_plan ||
      (*scale_plan)->summary.transient_bytes != scale_owners * scale_bytes ||
      (*scale_plan)->summary.allocation_count != scale_owners ||
      (*scale_plan)->summary.reuse_count != 0u ||
      (*scale_plan)->summary.largest_bytes != scale_bytes ||
      (*scale_plan)->chunks.size() != scale_owners) {
    return 19;
  }
  for (std::size_t index = 0u; index < scale_owners; ++index) {
    if ((*scale_plan)->chunks[index] != scale_words ||
        (*scale_plan)->owners[index] != index ||
        (*scale_plan)->offsets[index] != 0u) {
      return 20;
    }
  }
  return 0;
}

} // namespace rund_node_test_pipeline
