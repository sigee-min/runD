#pragma once

#include <rund/compute.hpp>

#include <array>
#include <concepts>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace package_compute {

using BatchJob = rund::compute::Job<std::int32_t(std::int32_t)>;

static_assert(rund::compute::Batch::capacity() == 64u);
static_assert(std::is_default_constructible_v<rund::compute::Batch>);
static_assert(!std::is_copy_constructible_v<rund::compute::Batch>);
static_assert(std::is_nothrow_move_constructible_v<rund::compute::Batch>);
static_assert(requires(rund::compute::Batch &batch, BatchJob &job) {
  { batch.add(job) } -> std::same_as<rund::compute::Status>;
  { batch.run() } -> std::same_as<rund::compute::Status>;
  { batch.size() } -> std::same_as<std::size_t>;
  { batch.stats() } -> std::same_as<rund::compute::Stats>;
});

inline int BatchRun() {
  rund::compute::Batch empty{};
  const auto empty_result = empty.run();
  if (empty_result ||
      empty_result.reason() != rund::compute::Reason::BatchEmpty ||
      empty_result.error() != "compute_batch_empty" ||
      empty_result.exit_code() != 1) {
    return 2;
  }

  constexpr std::array<std::int32_t, 4u> input{1, 2, 3, 4};
  auto cpu_program =
      rund::compute::on(rund::compute::Target::cpu())
          .map<std::int32_t>("batch-cpu", input.size(),
                             [](auto value) { return value + 1; })
          .compile();
  auto cpu_job =
      cpu_program
          ? cpu_program->resident(input)
          : decltype(cpu_program->resident(input))::fail(cpu_program.reason());
  if (!cpu_job) {
    return cpu_job.exit_code();
  }
  rund::compute::Batch cpu{};
  const auto cpu_result = cpu.add(*cpu_job);
  if (cpu_result ||
      cpu_result.reason() != rund::compute::Reason::BatchCpuUnsupported ||
      cpu_result.error() != "compute_batch_cpu_unsupported") {
    return 2;
  }

  constexpr std::array targets{rund::compute::Target::metal(),
                               rund::compute::Target::vulkan()};
  for (const rund::compute::Target target : targets) {
    auto device = rund::compute::open(target);
    if (!device) {
      if (device.code() != rund::compute::Code::Unavailable &&
          device.code() != rund::compute::Code::Unsupported) {
        return device.exit_code();
      }
      continue;
    }
    auto first_program =
        rund::compute::on(*device)
            .map<std::int32_t>("batch-first", input.size(),
                               [](auto value) { return value * 2 + 1; })
            .compile();
    auto second_program =
        rund::compute::on(*device)
            .map<std::int32_t>("batch-second", input.size(),
                               [](auto value) { return value + 7; })
            .compile();
    if (!first_program) {
      return first_program.exit_code();
    }
    if (!second_program) {
      return second_program.exit_code();
    }
    auto first = first_program->resident(input);
    auto second = second_program->resident(input);
    if (!first) {
      return first.exit_code();
    }
    if (!second) {
      return second.exit_code();
    }

    rund::compute::Batch batch{};
    const auto first_added = batch.add(*first);
    const auto second_added = batch.add(*second);
    if (!first_added) {
      return first_added.exit_code();
    }
    if (!second_added) {
      return second_added.exit_code();
    }
    const auto ran = batch.run();
    if (!ran) {
      return ran.exit_code();
    }
    const auto first_run = first->stats();
    const auto second_run = second->stats();
    if (batch.stats().command_submits != 1u ||
        first_run.command_submits != 0u || second_run.command_submits != 0u) {
      return 2;
    }
    const auto first_output = first->read();
    const auto second_output = second->read();
    if (!first_output) {
      return first_output.exit_code();
    }
    if (!second_output) {
      return second_output.exit_code();
    }
    if (*first_output != std::vector<std::int32_t>{3, 5, 7, 9} ||
        *second_output != std::vector<std::int32_t>{8, 9, 10, 11} ||
        first->stats().graph_hash == 0u || second->stats().graph_hash == 0u ||
        first->stats().output_hash == 0u || second->stats().output_hash == 0u) {
      return 2;
    }
  }
  return 0;
}

} // namespace package_compute
