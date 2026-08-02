#include "local.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/cpu/state.hpp"
#include "src/compute/job/local.hpp"
#include "src/compute/pipeline/state.hpp"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <utility>

namespace rund_node_test_pipeline {

[[nodiscard]] int CheckRepeat(rund::compute::Device &device,
                              const Backend backend) {
  using namespace rund::compute;
  constexpr std::array<std::int32_t, 4u> seed{1, 3, 5, 7};
  auto input = device.upload<std::int32_t>(std::span<const std::int32_t>{seed});
  auto output = device.buffer<std::int32_t>(seed.size());
  auto body = on(device)
                  .map<std::int32_t>("iterate", seed.size(),
                                     [](auto value) { return value + 1; })
                  .branch([](auto values) {
                    return values.map("iterate-finish",
                                      [](auto value) { return value + 1; });
                  })
                  .compile();
  if (!input || !output || !body) {
    return 1;
  }
  auto oversized = pipeline(device)
                       .repeat<PipelineIterationCapacity + 1u>(
                           *body, read(*input), write_final(*output))
                       .prepare();
  if (oversized || oversized.reason() != Reason::PipelineCapacity) {
    return 9;
  }
  auto loop = pipeline(device)
                  .profile(PipelineProfile::Steps)
                  .repeat<PipelineIterationCapacity>(*body, read(*input),
                                                     write_final(*output))
                  .prepare();
  if (!loop) {
    std::fprintf(stderr, "repeat prepare reason=%u\n",
                 static_cast<unsigned>(loop.reason()));
    return 2;
  }
  const std::shared_ptr<detail::PipelineState> &loop_state =
      detail::PipelineStateAccess::state(*loop);
  const std::shared_ptr<detail::ProgramState> &body_state =
      detail::ProgramAccess::state(*body);
  if (loop_state == nullptr || body_state == nullptr ||
      !CanonicalChunkOrder(*body_state) ||
      loop_state->steps.size() != PipelineIterationCapacity) {
    return 10;
  }
  std::array<const detail::JobState *, 3u> recurrence_owners{};
  std::size_t recurrence_owner_count = 0u;
  const bool needs_workspace = !body_state->chunks.empty();
  const detail::JobWorkspace *recurrence_workspace = nullptr;
  for (std::size_t iteration = 0u; iteration < loop_state->steps.size();
       ++iteration) {
    const detail::JobState *const owner =
        loop_state->steps[iteration].job.get();
    if (owner == nullptr) {
      return 10;
    }
    if (!owner->graph_buffers.empty() ||
        (needs_workspace && (owner->workspace == nullptr ||
                             owner->workspace->program != body_state ||
                             owner->workspace->offsets.size() !=
                                 owner->workspace->buffers.size())) ||
        (!needs_workspace && owner->workspace != nullptr)) {
      return 10;
    }
    for (std::size_t rank = 0u;
         needs_workspace && rank < body_state->chunk_order.size(); ++rank) {
      const std::size_t chunk = body_state->chunk_order[rank];
      if (chunk >= owner->workspace->buffers.size() ||
          owner->workspace->buffers[chunk] == nullptr ||
          (rank != 0u &&
           owner->workspace->buffers[chunk] !=
               owner->workspace->buffers[body_state->chunk_order.front()])) {
        return 10;
      }
    }
    if (needs_workspace) {
      if (recurrence_workspace == nullptr) {
        recurrence_workspace = owner->workspace.get();
      } else if (recurrence_workspace != owner->workspace.get()) {
        return 10;
      }
    }
    if (iteration >= 3u &&
        owner != loop_state->steps[iteration - 2u].job.get()) {
      return 10;
    }
    if (std::find(recurrence_owners.begin(),
                  recurrence_owners.begin() + recurrence_owner_count, owner) ==
        recurrence_owners.begin() + recurrence_owner_count) {
      if (recurrence_owner_count == recurrence_owners.size()) {
        return 10;
      }
      recurrence_owners[recurrence_owner_count++] = owner;
    }
  }
  if (recurrence_owner_count != recurrence_owners.size()) {
    return 10;
  }
  const Status repeated = loop->run();
  if (!repeated || loop->stats().pipeline_compiles != 0u) {
    std::fprintf(
        stderr, "repeat run reason=%u compiles=%llu\n",
        static_cast<unsigned>(repeated.reason()),
        static_cast<unsigned long long>(loop->stats().pipeline_compiles));
    return 2;
  }
  const Stats repeated_execution = loop->stats();
  std::array<std::int32_t, 4u> actual{};
  constexpr std::array<std::int32_t, 4u> expected{2049, 2051, 2053, 2055};
  std::array<PipelineStepProfile, PipelineIterationCapacity> rows{};
  const auto profile = loop->profile(rows);
  if (!ReadExact(*loop, *output, actual) || actual != expected ||
      loop->stats().pipeline.step_count != 1u ||
      loop->stats().pipeline.verified_step_count != 1u ||
      loop->stats().pipeline.failed_step_index !=
          PipelineStats::no_failed_step ||
      !profile || profile->total != rows.size() ||
      profile->written != rows.size() ||
      !ProfileMemoryReconciles(*profile, rows)) {
    return 3;
  }
  for (std::size_t iteration = 0u; iteration < rows.size(); ++iteration) {
    if (rows[iteration].index != 0u || rows[iteration].iteration != iteration ||
        rows[iteration].iteration_bound != rows.size()) {
      return 6;
    }
  }
  if (backend != Backend::Cpu) {
    const std::uint64_t read_submits = backend == Backend::Vulkan ? 1u : 0u;
    std::uint64_t profiled_dispatches = 0u;
    for (std::size_t iteration = 0u; iteration < rows.size(); ++iteration) {
      const PipelineStepProfile &row = rows[iteration];
      if (!row.execution.available() ||
          row.execution.original_dispatches == 0u ||
          (iteration == 0u ? row.execution.final_dispatches != 1u ||
                                 row.execution.workgroup_count == 0u ||
                                 row.execution.work_item_count != seed.size()
                           : row.execution.final_dispatches != 0u ||
                                 row.execution.workgroup_count != 0u ||
                                 row.execution.work_item_count != 0u ||
                                 row.timing.available())) {
        std::fprintf(
            stderr,
            "repeat profile backend=%u iteration=%zu available=%u "
            "original=%llu final=%llu groups=%llu items=%llu timing=%u\n",
            static_cast<unsigned>(backend), iteration,
            row.execution.available() ? 1u : 0u,
            static_cast<unsigned long long>(row.execution.original_dispatches),
            static_cast<unsigned long long>(row.execution.final_dispatches),
            static_cast<unsigned long long>(row.execution.workgroup_count),
            static_cast<unsigned long long>(row.execution.work_item_count),
            row.timing.available() ? 1u : 0u);
        return 11;
      }
      profiled_dispatches += row.execution.final_dispatches;
    }
    if (repeated_execution.dispatches != 1u ||
        repeated_execution.command_submits != 1u ||
        loop->stats().command_submits !=
            repeated_execution.command_submits + read_submits ||
        profiled_dispatches != 1u) {
      std::fprintf(
          stderr,
          "repeat aggregate backend=%u dispatches=%llu "
          "execution_submits=%llu observed_submits=%llu profiled=%llu\n",
          static_cast<unsigned>(backend),
          static_cast<unsigned long long>(repeated_execution.dispatches),
          static_cast<unsigned long long>(repeated_execution.command_submits),
          static_cast<unsigned long long>(loop->stats().command_submits),
          static_cast<unsigned long long>(profiled_dispatches));
      return 11;
    }

    auto plain_output = device.buffer<std::int32_t>(seed.size());
    auto plain = plain_output
                     ? pipeline(device)
                           .repeat<PipelineIterationCapacity>(
                               *body, read(*input), write_final(*plain_output))
                           .prepare()
                     : Result<Pipeline>::fail(Reason::PipelineInvalid);
    std::array<std::int32_t, seed.size()> plain_values{};
    const Status plain_status =
        plain ? plain->run() : Status::fail(plain.reason());
    const Stats plain_execution = plain ? plain->stats() : Stats{};
    const bool plain_read =
        plain && ReadExact(*plain, *plain_output, plain_values);
    if (!plain || !plain_status || !plain_read || plain_values != actual ||
        plain_execution.dispatches != repeated_execution.dispatches ||
        plain_execution.command_submits != repeated_execution.command_submits ||
        plain->stats().command_submits != loop->stats().command_submits) {
      std::fprintf(
          stderr,
          "repeat plain backend=%u prepare=%u run=%u read=%u exact=%u "
          "dispatches=%llu/%llu submits=%llu/%llu\n",
          static_cast<unsigned>(backend), plain ? 1u : 0u,
          plain_status ? 1u : 0u, plain_read ? 1u : 0u,
          plain_values == actual ? 1u : 0u,
          static_cast<unsigned long long>(plain_execution.dispatches),
          static_cast<unsigned long long>(repeated_execution.dispatches),
          static_cast<unsigned long long>(plain_execution.command_submits),
          static_cast<unsigned long long>(repeated_execution.command_submits));
      return 11;
    }
  }
  const auto first = loop->stats();
  if (!loop->run()) {
    return 4;
  }
  const auto warm = loop->stats();
  if (!WarmCountersClean(warm) || first.dispatches != warm.dispatches ||
      warm.pipeline.step_count != 1u ||
      warm.pipeline.verified_step_count != 1u ||
      warm.pipeline.failed_step_index != PipelineStats::no_failed_step) {
    return 5;
  }

  constexpr std::array<std::uint32_t, 1u> active_count{4u};
  auto count = device.upload<std::uint32_t>(active_count);
  auto active_values = device.upload<std::int32_t>(seed);
  auto remaining_values = device.buffer<std::int32_t>(seed.size());
  auto remaining_count = device.buffer<std::uint32_t>(1u);
  auto reference_count = device.upload<std::uint32_t>(active_count);
  auto reference_values = device.upload<std::int32_t>(seed);
  auto reference_remaining = device.buffer<std::int32_t>(seed.size());
  auto reference_remaining_count = device.buffer<std::uint32_t>(1u);
  auto active_body =
      on(device)
          .input<Bounded<std::int32_t>>(seed.size())
          .map("repeat-decay", [](auto value) { return value - 1; })
          .filter([](auto value) { return value > 0; })
          .compile();
  if (!count || !active_values || !remaining_values || !remaining_count ||
      !reference_count || !reference_values || !reference_remaining ||
      !reference_remaining_count || !active_body) {
    return 7;
  }
  auto active_loop =
      pipeline(device)
          .profile(PipelineProfile::Steps)
          .repeat<6u>(*active_body, read(*active_values, *count),
                      write_final(*remaining_values, *remaining_count))
          .prepare();
  auto reference_builder = pipeline(device).profile(PipelineProfile::Steps);
  reference_builder
      .then(*active_body, read(*reference_values, *reference_count),
            write(*reference_remaining, *reference_remaining_count))
      .then(*active_body,
            read(*reference_remaining, *reference_remaining_count),
            write(*reference_values, *reference_count))
      .then(*active_body, read(*reference_values, *reference_count),
            write(*reference_remaining, *reference_remaining_count))
      .then(*active_body,
            read(*reference_remaining, *reference_remaining_count),
            write(*reference_values, *reference_count))
      .then(*active_body, read(*reference_values, *reference_count),
            write(*reference_remaining, *reference_remaining_count))
      .then(*active_body,
            read(*reference_remaining, *reference_remaining_count),
            write(*reference_values, *reference_count));
  auto reference_loop = std::move(reference_builder).prepare();
  std::array<std::int32_t, 4u> remaining{};
  std::array<std::uint32_t, 1u> remaining_size{};
  std::array<std::int32_t, 4u> reference_output{};
  std::array<std::uint32_t, 1u> reference_size{};
  const Status active_status =
      active_loop ? active_loop->run() : Status::fail(active_loop.reason());
  const Status reference_status = reference_loop
                                      ? reference_loop->run()
                                      : Status::fail(reference_loop.reason());
  if (!active_loop || !reference_loop || !active_status || !reference_status ||
      !ReadExact(*active_loop, *remaining_values, remaining) ||
      !ReadExact(*active_loop, *remaining_count, remaining_size) ||
      !ReadExact(*reference_loop, *reference_values, reference_output) ||
      !ReadExact(*reference_loop, *reference_count, reference_size) ||
      remaining_size[0] != 1u || remaining[0] != 1 ||
      remaining_size != reference_size || remaining[0] != reference_output[0] ||
      active_loop->stats().pipeline.step_count != 1u ||
      active_loop->stats().pipeline.control_command_count !=
          reference_loop->stats().pipeline.control_command_count ||
      !SameControlStats(active_loop->stats().control,
                        reference_loop->stats().control)) {
    const Stats active_stats = active_loop ? active_loop->stats() : Stats{};
    const Stats reference_stats =
        reference_loop ? reference_loop->stats() : Stats{};
    std::fprintf(stderr,
                 "repeat bounded active=%u reference=%u active_prepare=%u "
                 "reference_prepare=%u active_count=%u reference_count=%u "
                 "active_commands=%llu reference_commands=%llu "
                 "active_generated=%llu reference_generated=%llu\n",
                 static_cast<unsigned>(active_status.reason()),
                 static_cast<unsigned>(reference_status.reason()),
                 static_cast<unsigned>(active_loop.reason()),
                 static_cast<unsigned>(reference_loop.reason()),
                 remaining_size[0u], reference_size[0u],
                 static_cast<unsigned long long>(
                     active_stats.pipeline.control_command_count),
                 static_cast<unsigned long long>(
                     reference_stats.pipeline.control_command_count),
                 static_cast<unsigned long long>(
                     active_stats.control.generated_item_count),
                 static_cast<unsigned long long>(
                     reference_stats.control.generated_item_count));
    return 8;
  }
  std::array<PipelineStepProfile, 6u> active_rows{};
  std::array<PipelineStepProfile, 6u> reference_rows{};
  const auto active_profile = active_loop->profile(active_rows);
  const auto reference_profile = reference_loop->profile(reference_rows);
  if (!active_profile || !reference_profile ||
      active_profile->written != active_rows.size() ||
      reference_profile->written != reference_rows.size()) {
    std::fprintf(stderr, "repeat bounded profile unavailable\n");
    return 8;
  }
  for (std::size_t iteration = 0u; iteration < active_rows.size();
       ++iteration) {
    if (!SameControlStats(active_rows[iteration].execution.control,
                          reference_rows[iteration].execution.control)) {
      std::fprintf(stderr, "repeat bounded profile mismatch iteration=%zu\n",
                   iteration);
      return 8;
    }
  }
  if (!active_loop->run()) {
    std::fprintf(stderr, "repeat bounded warm run failed\n");
    return 8;
  }
  const ControlStats stable_control = active_loop->stats().control;
  const std::uint64_t stable_control_commands =
      active_loop->stats().pipeline.control_command_count;
  const auto stable_profile = active_loop->profile(active_rows);
  const std::array<PipelineStepProfile, 6u> stable_rows = active_rows;
  if (!stable_profile || stable_profile->written != active_rows.size() ||
      !active_loop->run()) {
    std::fprintf(stderr, "repeat bounded stable profile failed\n");
    return 8;
  }
  const auto warm_profile = active_loop->profile(active_rows);
  if (!warm_profile || warm_profile->written != active_rows.size() ||
      active_loop->stats().pipeline.control_command_count !=
          stable_control_commands ||
      !SameControlStats(stable_control, active_loop->stats().control)) {
    const ControlStats current = active_loop->stats().control;
    std::fprintf(
        stderr,
        "repeat bounded warm control mismatch generated=%llu/%llu "
        "capacity=%llu/%llu dispatch=%llu/%llu work=%llu/%llu "
        "iterations=%llu/%llu skipped=%llu/%llu conflict=%llu/%llu "
        "overflow=%llu/%llu\n",
        static_cast<unsigned long long>(stable_control.generated_item_count),
        static_cast<unsigned long long>(current.generated_item_count),
        static_cast<unsigned long long>(stable_control.generated_capacity),
        static_cast<unsigned long long>(current.generated_capacity),
        static_cast<unsigned long long>(stable_control.indirect_dispatch_count),
        static_cast<unsigned long long>(current.indirect_dispatch_count),
        static_cast<unsigned long long>(
            stable_control.indirect_work_item_count),
        static_cast<unsigned long long>(current.indirect_work_item_count),
        static_cast<unsigned long long>(stable_control.iteration_count),
        static_cast<unsigned long long>(current.iteration_count),
        static_cast<unsigned long long>(stable_control.skipped_iteration_count),
        static_cast<unsigned long long>(current.skipped_iteration_count),
        static_cast<unsigned long long>(stable_control.conflict_count),
        static_cast<unsigned long long>(current.conflict_count),
        static_cast<unsigned long long>(stable_control.overflow_ordinal),
        static_cast<unsigned long long>(current.overflow_ordinal));
    return 8;
  }
  for (std::size_t iteration = 0u; iteration < active_rows.size();
       ++iteration) {
    if (!SameControlStats(stable_rows[iteration].execution.control,
                          active_rows[iteration].execution.control)) {
      std::fprintf(stderr,
                   "repeat bounded warm profile mismatch iteration=%zu\n",
                   iteration);
      return 8;
    }
  }

  // The second logical invocation reaches an empty workset after one body and
  // skips the next controlled body. It reuses the first invocation's external
  // output owners, so the empty result proves that earlier active bytes cannot
  // survive a skipped warm-history path.
  auto controlled =
      on(device)
          .input<std::int32_t>(seed.size())
          .branch([](auto values) {
            auto active = values.filter(
                [](auto value) { return value > std::int32_t{0}; });
            return active.template unroll<2u>(
                [](auto work) {
                  return work.map("repeat-reset-step", [](auto value) {
                    return value - std::int32_t{1};
                  });
                },
                [](auto value) { return value <= std::int32_t{0}; });
          })
          .compile();
  constexpr std::array<std::int32_t, 4u> live_input{4, 3, 0, 0};
  constexpr std::array<std::int32_t, 4u> empty_input{1, 0, 0, 0};
  constexpr std::array<std::int32_t, 4u> poisoned_values{91, 92, 93, 94};
  constexpr std::array<std::uint32_t, 1u> poisoned_count{95u};
  auto live_source = device.upload<std::int32_t>(live_input);
  auto empty_source = device.upload<std::int32_t>(empty_input);
  auto controlled_values = device.upload<std::int32_t>(poisoned_values);
  auto controlled_count = device.upload<std::uint32_t>(poisoned_count);
  if (!controlled || !live_source || !empty_source || !controlled_values ||
      !controlled_count) {
    return 8;
  }
  auto live = pipeline(device)
                  .then(*controlled, read(*live_source),
                        write(*controlled_values, *controlled_count))
                  .prepare();
  const auto controlled_state = detail::ProgramAccess::state(*controlled);
  if (controlled_state == nullptr ||
      controlled_state->graph_info.memory.reset_count != 2u ||
      controlled_state->graph_info.memory.reset_bytes !=
          2u * seed.size() * sizeof(std::int32_t)) {
    return 8;
  }
  std::array<std::size_t, 2u> reset_resources{};
  std::size_t reset_count = 0u;
  for (std::size_t index = 0u;
       index < controlled_state->graph_info.resources.size(); ++index) {
    if (!controlled_state->graph_info.resources[index].requires_reset()) {
      continue;
    }
    if (reset_count >= reset_resources.size()) {
      return 8;
    }
    reset_resources[reset_count++] = index;
  }
  if (reset_count != reset_resources.size()) {
    return 8;
  }
  const graph::Resource first_reset =
      controlled_state->graph_info.resources[reset_resources[0u]];
  const graph::Resource second_reset =
      controlled_state->graph_info.resources[reset_resources[1u]];
  const detail::GraphValueRoute first_route =
      controlled_state->graph_value_routes[reset_resources[0u]];
  const detail::GraphValueRoute second_route =
      controlled_state->graph_value_routes[reset_resources[1u]];
  if (first_route.source != detail::GraphBindSource::Internal ||
      second_route.source != detail::GraphBindSource::Internal ||
      first_route.index != second_route.index ||
      first_route.offset_bytes != second_route.offset_bytes ||
      first_reset.last_use >= second_reset.reset_node) {
    return 8;
  }
  std::array<std::int32_t, 4u> controlled_output{};
  std::array<std::uint32_t, 1u> controlled_size{};
  if (!live || !live->run() ||
      !ReadExact(*live, *controlled_values, controlled_output) ||
      !ReadExact(*live, *controlled_count, controlled_size) ||
      controlled_size[0u] != 2u ||
      controlled_output != std::array<std::int32_t, 4u>{2, 1, 0, 0}) {
    std::fprintf(
        stderr,
        "repeat live backend=%u prepare=%u count=%u values=%d,%d,%d,%d\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(live.reason()),
        controlled_size[0u], controlled_output[0u], controlled_output[1u],
        controlled_output[2u], controlled_output[3u]);
    return 8;
  }
  auto empty = pipeline(device)
                   .then(*controlled, read(*empty_source),
                         write(*controlled_values, *controlled_count))
                   .prepare();
  if (!empty || !empty->run() ||
      !ReadExact(*empty, *controlled_values, controlled_output) ||
      !ReadExact(*empty, *controlled_count, controlled_size) ||
      controlled_size[0u] != 0u ||
      controlled_output != std::array<std::int32_t, 4u>{0, 0, 0, 0} ||
      empty->stats().control.iteration_count != 1u ||
      empty->stats().control.skipped_iteration_count != 1u ||
      empty->stats().reset_bytes == 0u || empty->stats().reset_commands == 0u) {
    std::fprintf(
        stderr,
        "repeat reset backend=%u prepare=%u count=%u values=%d,%d,%d,%d "
        "iterations=%llu skipped=%llu reset_bytes=%llu reset_commands=%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(empty.reason()),
        controlled_size[0u], controlled_output[0u], controlled_output[1u],
        controlled_output[2u], controlled_output[3u],
        static_cast<unsigned long long>(
            empty ? empty->stats().control.iteration_count : 0u),
        static_cast<unsigned long long>(
            empty ? empty->stats().control.skipped_iteration_count : 0u),
        static_cast<unsigned long long>(empty ? empty->stats().reset_bytes
                                              : 0u),
        static_cast<unsigned long long>(empty ? empty->stats().reset_commands
                                              : 0u));
    return 8;
  }
  return 0;
}

} // namespace rund_node_test_pipeline
