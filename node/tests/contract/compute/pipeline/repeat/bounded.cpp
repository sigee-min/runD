#include "local.hpp"

#include <node/runtime/compute/access.hpp>

#include <cstdio>
#include <utility>

namespace rund_node_test_pipeline {

[[nodiscard]] int CheckBoundedRepeat(rund::compute::Device &device) {
  using namespace rund::compute;
  constexpr std::array<std::int32_t, 4u> seed{1, 3, 5, 7};
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

  return 0;
}

} // namespace rund_node_test_pipeline
