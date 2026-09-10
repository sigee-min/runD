#include "repeat/local.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/job/local.hpp"
#include "src/compute/pipeline/state.hpp"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <utility>

namespace rund_node_test_pipeline {
namespace {

[[nodiscard]] int CheckFixedRepeat(rund::compute::Device &device,
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
      !body_state->chunks.empty() || !CanonicalChunkOrder(*body_state) ||
      loop_state->steps.size() != PipelineIterationCapacity) {
    return 10;
  }
  std::array<const detail::JobState *, 3u> recurrence_owners{};
  std::size_t recurrence_owner_count = 0u;
  for (std::size_t iteration = 0u; iteration < loop_state->steps.size();
       ++iteration) {
    const detail::JobState *const owner =
        loop_state->steps[iteration].job.get();
    if (owner == nullptr) {
      return 10;
    }
    if (!owner->graph_buffers.empty() || owner->workspace != nullptr) {
      return 10;
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
        loop->stats().command_submits != repeated_execution.command_submits ||
        loop->stats().transfer_submissions.device_to_host != read_submits ||
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
        plain->stats().command_submits != loop->stats().command_submits ||
        plain->stats().transfer_submissions.device_to_host !=
            loop->stats().transfer_submissions.device_to_host) {
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

  return 0;
}

} // namespace

[[nodiscard]] int CheckRepeat(rund::compute::Device &device,
                              const Backend backend) {
  if (const int fixed = CheckFixedRepeat(device, backend); fixed != 0) {
    return fixed;
  }
  if (const int bounded = CheckBoundedRepeat(device); bounded != 0) {
    return bounded;
  }
  return CheckResetRepeat(device, backend);
}

} // namespace rund_node_test_pipeline
