#include "local.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/job/control/model.hpp"
#include "src/compute/job/local.hpp"
#include "src/compute/memory/cpu.hpp"
#include "src/compute/pipeline/state.hpp"
#include "src/compute/status.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <vector>

namespace rund_node_test_pipeline::memory {

[[nodiscard]] int CheckReset(rund::compute::Device &device) {
  using namespace rund::compute;
  const bool accelerated =
      detail::DeviceAccess::state(device)->backend != Backend::Cpu;

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
           partial_second_observed ==
               std::array<std::int32_t, 4u>{1, 4, 1, 5};
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
  if (!accelerated) {
    const auto zero = [](const MemoryCounter &counter) noexcept {
      return counter.current == 0u && counter.peak == 0u &&
             counter.cumulative == 0u && counter.reused == 0u &&
             counter.budget == 0u;
    };
    const PipelinePlan reset_plan = reset_shared->plan();
    const std::shared_ptr<detail::CpuPreparedArena> &cpu_arena =
        reset_state->cpu_prepared_arena;
    if (cpu_arena == nullptr || cpu_arena->payload_host_bytes() >
                                    std::numeric_limits<std::uint64_t>::max() -
                                        sizeof(detail::CpuPreparedArena)) {
      return 22;
    }
    std::uint64_t shared_cpu_host =
        sizeof(detail::CpuPreparedArena) + cpu_arena->payload_host_bytes();
    for (const std::shared_ptr<detail::CpuGraphStorage> &storage :
         reset_state->cpu_storage) {
      const detail::CpuStorageBytes memory =
          detail::cpu_graph_storage_private_memory(storage.get());
      if (memory.host >
          std::numeric_limits<std::uint64_t>::max() - shared_cpu_host) {
        shared_cpu_host = std::numeric_limits<std::uint64_t>::max();
        break;
      }
      shared_cpu_host += memory.host;
    }
    const bool rows_private =
        std::all_of(reset_rows.begin(), reset_rows.end(), [&](const auto &row) {
          return zero(row.memory.tile) &&
                 row.memory.host.current == sizeof(detail::JobState);
        });
    if (cpu_arena == nullptr || !rows_private ||
        reset_profile->shared_memory.tile.current !=
            reset_profile->memory.tile.current ||
        reset_profile->shared_memory.tile.current !=
            reset_plan.prepared_tile_bytes ||
        reset_profile->shared_memory.host.current < shared_cpu_host ||
        reset_state->cpu_storage.empty()) {
      return 22;
    }

    std::shared_ptr<detail::CpuPreparedArena> retained_arena =
        std::move(reset_state->cpu_storage.front()->prepared_arena);
    std::array<PipelineStepProfile, 2u> invalid_rows{};
    const auto invalid_profile = reset_shared->profile(invalid_rows);
    reset_state->cpu_storage.front()->prepared_arena = retained_arena;
    constexpr std::uint64_t saturated =
        std::numeric_limits<std::uint64_t>::max();
    if (!invalid_profile ||
        invalid_profile->shared_memory.host.current != saturated ||
        invalid_profile->shared_memory.tile.current != saturated ||
        invalid_profile->memory.host.current != saturated ||
        invalid_profile->memory.tile.current != saturated ||
        !ProfileMemoryReconciles(*invalid_profile, invalid_rows)) {
      return 23;
    }

    std::vector<std::shared_ptr<detail::CpuGraphStorage>> retained_storage;
    retained_storage.swap(reset_state->cpu_storage);
    std::array<PipelineStepProfile, 2u> missing_rows{};
    const auto missing_profile = reset_shared->profile(missing_rows);
    reset_state->cpu_storage.swap(retained_storage);
    if (!missing_profile ||
        missing_profile->shared_memory.host.current != saturated ||
        missing_profile->shared_memory.tile.current != saturated ||
        missing_profile->memory.host.current != saturated ||
        missing_profile->memory.tile.current != saturated ||
        !ProfileMemoryReconciles(*missing_profile, missing_rows)) {
      return 24;
    }
  }
  return 0;
}

} // namespace rund_node_test_pipeline::memory
