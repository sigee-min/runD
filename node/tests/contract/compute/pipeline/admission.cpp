#include "local.hpp"

#include "../allocation.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/pipeline/state.hpp"

#include <atomic>
#include <concepts>
#include <cstdint>
#include <limits>
#include <optional>
#include <thread>
#include <utility>

namespace rund_node_test_pipeline {
namespace {

using namespace rund::compute;

static_assert(requires(rund::Session &session, Target target, Compile compile,
                       DevicePipelineMemoryLimit limit) {
  { open(target, limit) } -> std::same_as<Result<Device>>;
  { open(target, compile, limit) } -> std::same_as<Result<Device>>;
  { open(session, target, limit) } -> std::same_as<Result<Device>>;
});

[[nodiscard]] Result<PipelineBuilder>
BuildPipeline(Device &device, const bool transactional,
              const LatestDeviceState *const restore = nullptr) {
  constexpr std::array<std::int32_t, 8u> input{1, 2, 3, 4, 5, 6, 7, 8};
  auto program =
      on(device)
          .map<std::int32_t>("device-pipeline-memory-admission", input.size(),
                             [](auto value) { return value + 1; })
          .compile();
  auto first = Upload(device, input);
  auto second = device.buffer<std::int32_t>(input.size());
  if (!program || !first || !second) {
    return Result<PipelineBuilder>::fail(Reason::PipelineInvalid);
  }
  PipelineBuilder builder = pipeline(device);
  if (transactional) {
    builder.state(*first, *second);
  }
  builder.then(*program, read(*first), write(*second));
  if (restore != nullptr) {
    builder.restore(*restore);
  }
  if (transactional) {
    builder.commit();
  }
  return Result<PipelineBuilder>::success(std::move(builder));
}

[[nodiscard]] Result<PipelinePlan> PlanPipeline(Device &device,
                                                const bool transactional) {
  auto built = BuildPipeline(device, transactional);
  if (!built) {
    return Result<PipelinePlan>::fail(built.reason());
  }
  return built->plan();
}

[[nodiscard]] bool Empty(const DevicePipelineMemoryReport &report,
                         const std::uint64_t capacity) noexcept {
  return report.available() && report.capacity_bytes == capacity &&
         report.committed_bytes == 0u && report.preparing_bytes == 0u &&
         report.available_bytes == capacity;
}

[[nodiscard]] int
CheckDefaultAndExactAdmission(const std::uint64_t committed_bytes,
                              const std::uint64_t peak_bytes) {
  auto unlimited_open = open(Target::cpu(1u));
  if (!unlimited_open) {
    return 1;
  }
  Device unlimited = std::move(unlimited_open).value();
  auto unlimited_builder = BuildPipeline(unlimited, false);
  if (!unlimited_builder) {
    return 2;
  }
  const auto unlimited_plan = unlimited_builder->plan();
  if (!unlimited_plan ||
      unlimited_plan->committed_peak_bytes != committed_bytes ||
      unlimited_plan->peak_bytes != peak_bytes) {
    return 3;
  }
  {
    auto prepared = std::move(unlimited_builder).value().prepare();
    const DevicePipelineMemoryReport report = unlimited.pipeline_memory();
    if (!prepared || !report.available() || report.limited() ||
        report.capacity_bytes != std::numeric_limits<std::uint64_t>::max() ||
        report.committed_bytes != committed_bytes ||
        report.preparing_bytes != 0u || report.admission_count != 1u) {
      return 4;
    }
  }
  if (!Empty(unlimited.pipeline_memory(),
             std::numeric_limits<std::uint64_t>::max())) {
    return 5;
  }

  auto exact_open =
      open(Target::cpu(1u), DevicePipelineMemoryLimit{committed_bytes});
  if (!exact_open) {
    return 6;
  }
  Device exact = std::move(exact_open).value();
  if (!exact.pipeline_memory().limited() ||
      exact.pipeline_memory().capacity_bytes != committed_bytes ||
      exact.memory().backend != Backend::Cpu) {
    return 7;
  }
  std::optional<Pipeline> held;
  auto first_builder = BuildPipeline(exact, false);
  if (!first_builder) {
    return 8;
  }
  auto first = std::move(first_builder).value().prepare();
  if (!first) {
    return 9;
  }
  held.emplace(std::move(first).value());
  const DevicePipelineMemoryReport full = exact.pipeline_memory();
  if (full.committed_bytes != committed_bytes || full.available_bytes != 0u ||
      full.preparing_bytes != 0u) {
    return 10;
  }
  auto second_builder = BuildPipeline(exact, false);
  if (!second_builder) {
    return 11;
  }
  const auto second = std::move(second_builder).value().prepare();
  if (second || second.reason() != Reason::DevicePipelineMemoryCapacity ||
      exact.pipeline_memory().rejection_count != 1u ||
      exact.pipeline_memory().committed_bytes != committed_bytes) {
    return 12;
  }
  held.reset();
  if (!Empty(exact.pipeline_memory(), committed_bytes)) {
    return 13;
  }
  auto retry_builder = BuildPipeline(exact, false);
  if (!retry_builder) {
    return 14;
  }
  {
    auto retry = std::move(retry_builder).value().prepare();
    if (!retry || exact.pipeline_memory().committed_bytes != committed_bytes) {
      return 15;
    }
  }
  return Empty(exact.pipeline_memory(), committed_bytes) ? 0 : 16;
}

[[nodiscard]] int
CheckPrecedenceAndRollback(const std::uint64_t committed_bytes,
                           const std::uint64_t peak_bytes) {
  auto short_open =
      open(Target::cpu(1u), DevicePipelineMemoryLimit{committed_bytes - 1u});
  if (!short_open) {
    return 1;
  }
  Device short_device = std::move(short_open).value();
  auto local_builder = BuildPipeline(short_device, false);
  if (!local_builder) {
    return 2;
  }
  local_builder->budget(MemoryBudget{peak_bytes - 1u});
  const auto local = std::move(local_builder).value().prepare();
  const DevicePipelineMemoryReport after_local = short_device.pipeline_memory();
  if (local || local.reason() != Reason::PipelineMemoryBudget ||
      after_local.admission_count != 0u || after_local.rejection_count != 0u ||
      !Empty(after_local, committed_bytes - 1u)) {
    return 3;
  }
  auto device_builder = BuildPipeline(short_device, false);
  if (!device_builder) {
    return 4;
  }
  const auto device_rejected = std::move(device_builder).value().prepare();
  const DevicePipelineMemoryReport after_device =
      short_device.pipeline_memory();
  if (device_rejected ||
      device_rejected.reason() != Reason::DevicePipelineMemoryCapacity ||
      after_device.admission_count != 0u ||
      after_device.rejection_count != 1u ||
      !Empty(after_device, committed_bytes - 1u)) {
    return 5;
  }

  auto rollback_open =
      open(Target::cpu(1u), DevicePipelineMemoryLimit{committed_bytes});
  if (!rollback_open) {
    return 6;
  }
  Device rollback = std::move(rollback_open).value();
  auto failed_builder = BuildPipeline(rollback, false);
  if (!failed_builder || !failed_builder->plan()) {
    return 7;
  }
  node_compute_allocation::FailNext();
  const auto failed = std::move(failed_builder).value().prepare();
  const DevicePipelineMemoryReport released = rollback.pipeline_memory();
  if (failed || failed.reason() != Reason::PipelineCapacity ||
      released.admission_count != 1u || released.release_count != 1u ||
      !Empty(released, committed_bytes)) {
    return 8;
  }
  auto retry_builder = BuildPipeline(rollback, false);
  if (!retry_builder) {
    return 9;
  }
  {
    const auto retry = std::move(retry_builder).value().prepare();
    if (!retry ||
        rollback.pipeline_memory().committed_bytes != committed_bytes) {
      return 10;
    }
  }
  return Empty(rollback.pipeline_memory(), committed_bytes) ? 0 : 11;
}

[[nodiscard]] int
CheckConcurrentAdmission(const std::uint64_t committed_bytes) {
  auto opened =
      open(Target::cpu(1u), DevicePipelineMemoryLimit{committed_bytes});
  if (!opened) {
    return 1;
  }
  Device device = std::move(opened).value();
  std::array<std::optional<PipelineBuilder>, 2u> builders;
  for (std::size_t index = 0u; index < builders.size(); ++index) {
    auto built = BuildPipeline(device, false);
    if (!built || !built->plan()) {
      return 2;
    }
    builders[index].emplace(std::move(built).value());
  }
  std::array<std::optional<Pipeline>, 2u> winners;
  std::array<Reason, 2u> reasons{Reason::Ok, Reason::Ok};
  std::atomic<std::uint32_t> ready{0u};
  std::atomic<bool> start{false};
  std::array<std::thread, 2u> threads;
  for (std::size_t index = 0u; index < threads.size(); ++index) {
    threads[index] = std::thread{[&, index] {
      ready.fetch_add(1u, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      auto prepared = std::move(*builders[index]).prepare();
      if (prepared) {
        winners[index].emplace(std::move(prepared).value());
      } else {
        reasons[index] = prepared.reason();
      }
    }};
  }
  while (ready.load(std::memory_order_acquire) != threads.size()) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  for (std::thread &thread : threads) {
    thread.join();
  }
  const std::size_t accepted =
      static_cast<std::size_t>(winners[0u].has_value()) +
      static_cast<std::size_t>(winners[1u].has_value());
  const bool rejected_reason =
      (!winners[0u] && reasons[0u] == Reason::DevicePipelineMemoryCapacity) ||
      (!winners[1u] && reasons[1u] == Reason::DevicePipelineMemoryCapacity);
  const DevicePipelineMemoryReport full = device.pipeline_memory();
  if (accepted != 1u || !rejected_reason ||
      full.committed_bytes != committed_bytes || full.preparing_bytes != 0u ||
      full.admission_count != 1u || full.rejection_count != 1u) {
    return 3;
  }
  winners[0u].reset();
  winners[1u].reset();
  return Empty(device.pipeline_memory(), committed_bytes) ? 0 : 4;
}

[[nodiscard]] int
CheckPublicationLifetime(const std::uint64_t transactional_bytes) {
  if (transactional_bytes > std::numeric_limits<std::uint64_t>::max() / 2u) {
    return 1;
  }
  const std::uint64_t capacity = transactional_bytes * 2u;
  auto opened = open(Target::cpu(1u), DevicePipelineMemoryLimit{capacity});
  if (!opened) {
    return 2;
  }
  Device device = std::move(opened).value();
  constexpr std::array<std::int32_t, 8u> input{1, 2, 3, 4, 5, 6, 7, 8};
  auto program =
      on(device)
          .map<std::int32_t>("device-pipeline-memory-admission", input.size(),
                             [](auto value) { return value + 1; })
          .compile();
  auto first = Upload(device, input);
  auto second = device.buffer<std::int32_t>(input.size());
  if (!program || !first || !second) {
    return 3;
  }
  auto source_builder = pipeline(device)
                            .state(*first, *second)
                            .then(*program, read(*first), write(*second))
                            .commit();
  const auto source_plan = source_builder.plan();
  if (!source_plan ||
      source_plan->committed_peak_bytes != transactional_bytes) {
    return 4;
  }
  auto source_result = std::move(source_builder).prepare();
  if (!source_result) {
    return 5;
  }
  std::optional<Pipeline> source;
  source.emplace(std::move(source_result).value());
  const detail::PipelineState *const source_state =
      detail::PipelineStateAccess::state(*source).get();
  auto latest_result = source->latest_device_state();
  if (source_state == nullptr || source_state->publication == nullptr ||
      !latest_result) {
    return 6;
  }
  const std::uint64_t publication_bytes =
      source_state->publication->publication_memory.usage().allocated_bytes;
  const detail::PipelinePublicationState *const publication =
      source_state->publication.get();
  if (publication_bytes == 0u ||
      !source_state->publication->publication_memory.committed() ||
      source_state->private_memory.usage().allocated_bytes !=
          transactional_bytes - publication_bytes ||
      device.pipeline_memory().committed_bytes != transactional_bytes) {
    return 7;
  }
  std::optional<LatestDeviceState> latest;
  latest.emplace(std::move(latest_result).value());

  auto peer_builder = pipeline(device)
                          .state(*first, *second)
                          .then(*program, read(*first), write(*second))
                          .restore(*latest)
                          .commit();
  auto peer = std::move(peer_builder).prepare();
  if (!peer) {
    return 8;
  }
  std::optional<Pipeline> peer_owner;
  peer_owner.emplace(std::move(peer).value());
  const detail::PipelineState *const peer_state =
      detail::PipelineStateAccess::state(*peer_owner).get();
  const std::uint64_t shared_committed =
      transactional_bytes * 2u - publication_bytes;
  if (peer_state == nullptr || peer_state->publication.get() != publication ||
      peer_state->publication->publication_memory.usage().allocated_bytes !=
          publication_bytes ||
      device.pipeline_memory().committed_bytes != shared_committed) {
    return 9;
  }
  source.reset();
  if (device.pipeline_memory().committed_bytes != transactional_bytes) {
    return 10;
  }
  peer_owner.reset();
  if (device.pipeline_memory().committed_bytes != publication_bytes) {
    return 11;
  }
  latest.reset();
  return Empty(device.pipeline_memory(), capacity) ? 0 : 12;
}

} // namespace

[[nodiscard]] int CheckDevicePipelineMemoryAdmission() {
  using namespace rund::compute;
  const auto zero =
      open(Target::cpu(1u), DevicePipelineMemoryLimit{.bytes = 0u});
  if (zero || zero.reason() != Reason::DevicePipelineMemoryCapacity) {
    return 1;
  }
  auto probe_open = open(Target::cpu(1u));
  if (!probe_open) {
    return 2;
  }
  Device probe = std::move(probe_open).value();
  const auto ordinary = PlanPipeline(probe, false);
  const auto repeated_ordinary = PlanPipeline(probe, false);
  const auto transactional = PlanPipeline(probe, true);
  if (!ordinary || !repeated_ordinary || !transactional ||
      ordinary->arena_extent_bytes == 0u || ordinary->peak_bytes <= 1u ||
      ordinary->committed_peak_bytes < ordinary->peak_bytes ||
      ordinary->peak_bytes != repeated_ordinary->peak_bytes ||
      ordinary->arena_extent_bytes != repeated_ordinary->arena_extent_bytes ||
      ordinary->committed_peak_bytes !=
          repeated_ordinary->committed_peak_bytes ||
      transactional->committed_peak_bytes <= 1u) {
    return 3;
  }
  if (const int exact = CheckDefaultAndExactAdmission(
          ordinary->committed_peak_bytes, ordinary->peak_bytes);
      exact != 0) {
    return 100 + exact;
  }
  if (const int precedence = CheckPrecedenceAndRollback(
          ordinary->committed_peak_bytes, ordinary->peak_bytes);
      precedence != 0) {
    return 200 + precedence;
  }
  if (const int concurrent =
          CheckConcurrentAdmission(ordinary->committed_peak_bytes);
      concurrent != 0) {
    return 300 + concurrent;
  }
  if (const int publication =
          CheckPublicationLifetime(transactional->committed_peak_bytes);
      publication != 0) {
    return 400 + publication;
  }
  return 0;
}

} // namespace rund_node_test_pipeline
