#include "../../support.hpp"
#include "local.hpp"

#include <rund/compute.hpp>
#include <rund/compute/session.hpp>
#include <rund/task/api.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <vector>

namespace rund::node::test_contract {

int CheckComputeAccelBackend(const compute::Target target) {
  const compute::Backend backend = target.backend();
  const std::array<std::int32_t, 8> input{-3, 0, 4, 9, 12, -8, 7, 21};
  auto program =
      compute::on(target)
          .map<std::int32_t>("node-host-accel", input.size(),
                             [](auto value) { return value * 2 + 5; })
          .compile();
  if (!program) {
    std::fprintf(stderr, "node host accel backend=%u compile reason=%.*s\n",
                 static_cast<unsigned>(backend),
                 static_cast<int>(program.error().size()),
                 program.error().data());
    return 1;
  }
  auto job = program->resident(input);
  if (!job) {
    return 2;
  }

  TelemetryProbe telemetry{};
  ::rund::Session session{};
  if (!session.open(Options(&telemetry))) {
    return 3;
  }
  compute::Submission task{};
  std::atomic_bool progressed{false};
  rund::task::Handle peer{};
  rund::task::Status peer_join{};
  const auto scope = session.scope([&] {
    task = session.compute(*job).submit();
    peer = rund::task::spawn("compute-peer", [&] {
      progressed.store(true, std::memory_order_release);
    });
    peer_join = rund::task::join(peer);
  });
  if (!scope || !progressed.load(std::memory_order_acquire)) {
    std::fprintf(
        stderr, "node host accel backend=%u scope=%.*s peer=%.*s join=%.*s\n",
        static_cast<unsigned>(backend), static_cast<int>(scope.error().size()),
        scope.error().data(), static_cast<int>(peer.error().size()),
        peer.error().data(), static_cast<int>(peer_join.error().size()),
        peer_join.error().data());
    return 4;
  }
  if (scope.tasks().external_parks() == 0u ||
      scope.tasks().external_parks() != scope.tasks().external_wakes()) {
    return 8;
  }
  const auto result = task.wait();
  if (!result) {
    std::fprintf(stderr, "node host accel backend=%u reason=%.*s\n",
                 static_cast<unsigned>(backend),
                 static_cast<int>(result.error().size()),
                 result.error().data());
    return 5;
  }
  const compute::Stats warm = result.stats();
  if (warm.backend != backend || warm.pipeline_compiles != 0u ||
      warm.buffer_allocations != 0u || warm.uploaded_bytes != 0u ||
      warm.download_events != 0u || warm.dispatches == 0u) {
    return 6;
  }
  if (telemetry.events != 1u || telemetry.event.compute.dispatches != 1u ||
      !telemetry.event.error().empty() ||
      telemetry.event.replay.code != rund::replay::Code::Ok) {
    return 10;
  }
  auto output = job->read();
  if (!output ||
      *output != std::vector<std::int32_t>{-1, 5, 13, 23, 29, -11, 19, 47}) {
    return 7;
  }
  const std::vector<std::int32_t> empty;
  auto empty_program =
      compute::on(target)
          .map<std::int32_t>("node-host-accel-empty", 0u,
                             [](auto value) { return value + 1; })
          .compile();
  if (!empty_program) {
    return 40;
  }
  auto empty_job = empty_program->resident(empty);
  if (!empty_job) {
    return 41;
  }
  compute::Submission empty_task{};
  const auto empty_scope =
      session.scope([&] { empty_task = session.compute(*empty_job).submit(); });
  const compute::Poll submitted_poll = empty_task.poll();
  if (!empty_scope || !submitted_poll.submitted) {
    return 42;
  }
  const compute::Completion empty_result = empty_task.wait();
  const compute::Poll completed_poll = empty_task.poll();
  const compute::Stats empty_stats = empty_result.stats();
  if (!empty_result || !completed_poll.submitted || !completed_poll.completed ||
      empty_stats.backend != backend || empty_stats.graph_hash == 0u ||
      empty_stats.pipeline_compiles != 0u ||
      empty_stats.buffer_allocations != 0u ||
      empty_stats.download_events != 0u) {
    return 43;
  }
  const auto empty_output = empty_job->read();
  if (!empty_output || !empty_output->empty() ||
      empty_job->stats().output_hash == 0u) {
    return 44;
  }
  auto reset_program =
      compute::on(target)
          .map<std::uint32_t>("node-host-accel-reset", 1u,
                              [](auto value) { return value; })
          .scatter(1u, {.count = 2u})
          .compile();
  constexpr std::array<std::uint32_t, 1u> reset_value{7u};
  constexpr std::array<std::uint32_t, 1u> reset_index{0u};
  if (!reset_program) {
    return 45;
  }
  auto reset_job = reset_program->resident(reset_value, reset_index);
  if (!reset_job || !session.compute(*reset_job).submit().wait()) {
    return 46;
  }
  auto reset_output = reset_job->read();
  constexpr std::array<std::uint32_t, 2u> first_reset{7u, 0u};
  if (!reset_output ||
      !std::equal(reset_output->begin(), reset_output->end(),
                  first_reset.begin(), first_reset.end()) ||
      reset_job->stats().reset_bytes != 2u * sizeof(std::uint32_t) ||
      reset_job->stats().reset_commands != 1u) {
    return 47;
  }
  constexpr std::array<std::uint32_t, 1u> next_reset_value{9u};
  constexpr std::array<std::uint32_t, 1u> next_reset_index{1u};
  if (!reset_job->write(next_reset_value, next_reset_index) ||
      !session.compute(*reset_job).submit().wait()) {
    return 48;
  }
  reset_output = reset_job->read();
  constexpr std::array<std::uint32_t, 2u> second_reset{0u, 9u};
  if (!reset_output ||
      !std::equal(reset_output->begin(), reset_output->end(),
                  second_reset.begin(), second_reset.end()) ||
      reset_job->stats().reset_bytes != 2u * sizeof(std::uint32_t) ||
      reset_job->stats().reset_commands != 1u) {
    return 49;
  }
  const std::array<std::int32_t, 8> other_input{8, 7, 6, 5, 4, 3, 2, 1};
  if (const int concurrency = CheckComputeAccelConcurrency(
          session, target, *program, input, other_input);
      concurrency != 0) {
    return 10 + concurrency;
  }
  if (backend == compute::Backend::Vulkan) {
    if (const int capacity = CheckVulkanCommandCapacity(*program, input);
        capacity != 0) {
      return 100 + capacity;
    }
  }
  if (const int retained = CheckComputeAccelLifetime(session, target);
      retained != 0) {
    return 10 + retained;
  }
  if (job->stats().download_events != 1u || !session.close()) {
    return 9;
  }
  return 0;
}

} // namespace rund::node::test_contract
