#include "local.hpp"

#include "../allocation.hpp"

#include <cstdio>
#include <utility>

namespace rund_node_test_pipeline {

[[nodiscard]] int CheckProfile(rund::compute::Device &device,
                               const Backend backend) {
  using namespace rund::compute;
  constexpr std::array<std::int32_t, 4u> input_values{1, 2, 3, 4};
  constexpr std::array<std::int32_t, 4u> expected_values{4, 6, 8, 10};
  auto increment =
      on(device)
          .map<std::int32_t>("pipeline-profile-increment", input_values.size(),
                             [](auto value) { return value + 1; })
          .compile();
  auto double_value =
      on(device)
          .map<std::int32_t>("pipeline-profile-double", input_values.size(),
                             [](auto value) { return value * 2; })
          .compile();
  auto input = Upload(device, input_values);
  auto middle = device.buffer<std::int32_t>(input_values.size());
  auto output = device.buffer<std::int32_t>(input_values.size());
  if (!increment || !double_value || !input || !middle || !output) {
    return 1;
  }

  auto unprofiled = pipeline(device)
                        .then(*increment, read(*input), write(*middle))
                        .then(*double_value, read(*middle), write(*output))
                        .prepare();
  std::array<PipelineStepProfile, 2u> unavailable_rows{};
  const auto unavailable =
      unprofiled
          ? unprofiled->profile(unavailable_rows)
          : Result<PipelineProfileSnapshot>::fail(Reason::PipelineInvalid);
  if (!unprofiled || unavailable ||
      unavailable.reason() != Reason::ProfileUnavailable) {
    return 2;
  }
  Pipeline moved{std::move(*unprofiled)};
  const auto invalid = unprofiled->profile(unavailable_rows);
  if (invalid || invalid.reason() != Reason::ProfileInvalid || !moved) {
    return 3;
  }

  auto profiled = pipeline(device)
                      .profile(PipelineProfile::Steps)
                      .then(*increment, read(*input), write(*middle))
                      .then(*double_value, read(*middle), write(*output))
                      .prepare();
  if (!profiled) {
    return 4;
  }
  std::array<PipelineStepProfile, 1u> cold_rows{};
  const auto cold = profiled->profile(cold_rows);
  if (!cold || cold->written != 1u || cold->total != 2u || !cold->truncated() ||
      cold_rows[0].index != 0u ||
      cold_rows[0].program != increment->fingerprint() ||
      cold_rows[0].execution.available() || cold_rows[0].timing.available() ||
      !cold->observation.available() ||
      cold->observation.clock != StepClock::HostSteady ||
      cold->observation.relation != StepTimingRelation::Exclusive ||
      cold->observation.sample_count != 1u) {
    return 5;
  }

  if (!profiled->run()) {
    return 6;
  }
  std::array<PipelineStepProfile, 2u> first_rows{};
  const auto first_profile = profiled->profile(first_rows);
  if (!first_profile || !ProfileMemoryReconciles(*first_profile, first_rows)) {
    return 7;
  }

  std::array<MemoryEntry, 128u> before_entries{};
  const MemorySnapshot before_memory =
      profiled->memory_snapshot(before_entries);
  std::array<PipelineStepProfile, 2u> warm_rows{};
  if (backend == Backend::Cpu) {
    node_compute_allocation::Start();
  }
  const Status warm_run = profiled->run();
  const auto warm_profile = profiled->profile(warm_rows);
  if (backend == Backend::Cpu) {
    node_compute_allocation::Stop();
  }
  std::array<MemoryEntry, 128u> after_entries{};
  const MemorySnapshot after_memory = profiled->memory_snapshot(after_entries);
  const bool retained_memory_stable =
      !before_memory.truncated() && !after_memory.truncated() &&
      before_memory.written == after_memory.written &&
      before_memory.total == after_memory.total &&
      SameMemoryEntries(std::span<const MemoryEntry>{before_entries.data(),
                                                     before_memory.written},
                        std::span<const MemoryEntry>{after_entries.data(),
                                                     after_memory.written}) &&
      SameMemory(before_memory.summary, after_memory.summary);
  const bool warm_stats_stable =
      backend == Backend::Cpu ||
      (warm_profile &&
       SameWarmStats(first_profile->execution, warm_profile->execution));
  const bool warm_counters_clean =
      warm_profile && WarmCountersClean(warm_profile->execution);
  if (!warm_run || !warm_profile ||
      (backend == Backend::Cpu && node_compute_allocation::Count() != 0u) ||
      !retained_memory_stable || !warm_stats_stable || !warm_counters_clean) {
    std::fprintf(
        stderr,
        "pipeline profile warm backend=%u run=%u profile=%u alloc=%llu "
        "memory=%u stats=%u counters=%u\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(warm_run.ok()),
        static_cast<unsigned>(warm_profile.ok()),
        static_cast<unsigned long long>(node_compute_allocation::Count()),
        static_cast<unsigned>(retained_memory_stable),
        static_cast<unsigned>(warm_stats_stable),
        static_cast<unsigned>(warm_counters_clean));
    return 8;
  }
  const auto &first_step = warm_rows[0];
  const auto &second_step = warm_rows[1];
  const bool host_timing =
      first_step.timing.available() && second_step.timing.available() &&
      first_step.timing.sample_count == 1u &&
      second_step.timing.sample_count == 1u &&
      first_step.timing.clock == StepClock::HostSteady &&
      second_step.timing.clock == StepClock::HostSteady &&
      first_step.timing.relation == StepTimingRelation::Exclusive &&
      second_step.timing.relation == StepTimingRelation::Exclusive;
  const bool device_timing =
      first_step.timing.available() && second_step.timing.available() &&
      first_step.timing.sample_count == 1u &&
      second_step.timing.sample_count == 1u &&
      first_step.timing.clock == StepClock::Device &&
      second_step.timing.clock == StepClock::Device &&
      first_step.timing.relation == StepTimingRelation::NonAdditive &&
      second_step.timing.relation == StepTimingRelation::NonAdditive;
  const bool timing_unavailable = TimingUnavailable(first_step.timing) &&
                                  TimingUnavailable(second_step.timing);
  constexpr std::uint64_t declared_step_count = 2u;
  constexpr std::uint64_t active_step_count = 2u;
  constexpr std::uint64_t declared_control_bytes =
      declared_step_count * sizeof(ControlStats);
  constexpr std::uint64_t timestamp_bytes =
      2u * active_step_count * sizeof(std::uint64_t);
  const bool cpu_instrumentation =
      warm_profile->instrumentation_command_count == 0u &&
      warm_profile->instrumentation_byte_count == 0u;
  const bool metal_instrumentation =
      warm_profile->instrumentation_command_count == 0u &&
      warm_profile->instrumentation_byte_count == declared_control_bytes;
  const bool vulkan_without_timestamps =
      warm_profile->instrumentation_command_count == 3u &&
      warm_profile->instrumentation_byte_count == declared_control_bytes;
  const bool vulkan_with_timestamps =
      warm_profile->instrumentation_command_count ==
          4u + 2u * active_step_count &&
      warm_profile->instrumentation_byte_count ==
          declared_control_bytes + timestamp_bytes;
  const bool backend_profile_valid =
      backend == Backend::Cpu     ? host_timing && cpu_instrumentation
      : backend == Backend::Metal ? timing_unavailable && metal_instrumentation
      : backend == Backend::Vulkan
          ? (timing_unavailable || device_timing) &&
                (vulkan_without_timestamps || vulkan_with_timestamps) &&
                (!device_timing || vulkan_with_timestamps)
          : false;
  const bool cpu_work =
      first_step.execution.worker_count == 2u &&
      second_step.execution.worker_count == 2u &&
      first_step.execution.participating_workers != 0u &&
      second_step.execution.participating_workers != 0u &&
      first_step.execution.tile_count != 0u &&
      second_step.execution.tile_count != 0u &&
      first_step.execution.vector_chunks + first_step.execution.tail_chunks !=
          0u &&
      second_step.execution.vector_chunks + second_step.execution.tail_chunks !=
          0u;
  const bool native_work =
      first_step.execution.workgroup_count == 1u &&
      second_step.execution.workgroup_count == 1u &&
      first_step.execution.work_item_count == input_values.size() &&
      second_step.execution.work_item_count == input_values.size();
  if (warm_profile->written != 2u || warm_profile->total != 2u ||
      warm_profile->truncated() || first_step.index != 0u ||
      second_step.index != 1u ||
      first_step.program != increment->fingerprint() ||
      second_step.program != double_value->fingerprint() ||
      !first_step.execution.available() || !second_step.execution.available() ||
      first_step.execution.sample_count != 1u ||
      second_step.execution.sample_count != 1u ||
      first_step.execution.original_dispatches != 1u ||
      second_step.execution.original_dispatches != 1u ||
      first_step.execution.final_dispatches != 1u ||
      second_step.execution.final_dispatches != 1u ||
      first_step.execution.barrier_count != 0u ||
      second_step.execution.barrier_count != 1u ||
      !(backend == Backend::Cpu ? cpu_work : native_work) ||
      !backend_profile_valid || first_step.memory.backend != backend ||
      second_step.memory.backend != backend ||
      first_step.memory.scope != MemoryScope::Pipeline ||
      second_step.memory.scope != MemoryScope::Pipeline ||
      warm_profile->shared_memory.backend != backend ||
      warm_profile->shared_memory.scope != MemoryScope::Pipeline ||
      warm_profile->execution.backend != backend ||
      warm_profile->execution.dispatches != 2u ||
      warm_profile->execution.command_submits !=
          (backend == Backend::Cpu ? 0u : 1u) ||
      warm_profile->execution.pipeline.step_count != 2u ||
      warm_profile->execution.pipeline.resource_count != 3u ||
      warm_profile->execution.pipeline.verified_step_count != 2u ||
      warm_profile->execution.pipeline.failed_step_index !=
          PipelineStats::no_failed_step ||
      warm_profile->execution.pipeline.control_byte_count !=
          (backend == Backend::Cpu ? 0u : 80u) ||
      warm_profile->referenced_resource_bytes != sizeof(input_values) * 3u ||
      !warm_profile->observation.available() ||
      warm_profile->observation.clock != StepClock::HostSteady ||
      warm_profile->observation.relation != StepTimingRelation::Exclusive ||
      warm_profile->observation.sample_count != 1u ||
      !ProfileMemoryReconciles(*warm_profile, warm_rows) ||
      !SameMemory(warm_profile->memory, profiled->memory())) {
    return 9;
  }
  std::array<PipelineStepProfile, 1u> truncated_rows{};
  const auto truncated = profiled->profile(truncated_rows);
  if (!truncated || truncated->written != 1u || truncated->total != 2u ||
      !truncated->truncated() || truncated_rows[0].index != 0u ||
      truncated_rows[0].program != increment->fingerprint() ||
      !SameMemory(truncated->memory, warm_profile->memory) ||
      !SameMemory(truncated->shared_memory, warm_profile->shared_memory) ||
      truncated->referenced_resource_bytes !=
          warm_profile->referenced_resource_bytes) {
    return 11;
  }
  std::array<std::int32_t, input_values.size()> observed{};
  if (!ReadExact(*profiled, *output, observed) || observed != expected_values) {
    return 10;
  }

  auto zero_program =
      on(device)
          .map<std::int32_t>("pipeline-profile-zero-work", 0u,
                             [](auto value) { return value + 1; })
          .compile();
  const std::array<std::int32_t, 0u> empty{};
  auto zero_input = Upload(device, empty);
  auto zero_output = device.buffer<std::int32_t>(0u);
  auto zero_pipeline =
      zero_program && zero_input && zero_output
          ? pipeline(device)
                .profile(PipelineProfile::Steps)
                .then(*zero_program, read(*zero_input), write(*zero_output))
                .prepare()
          : Result<Pipeline>::fail(Reason::PipelineInvalid);
  if (!zero_pipeline || !zero_pipeline->run()) {
    return 12;
  }
  std::array<PipelineStepProfile, 1u> zero_rows{};
  const auto zero_profile = zero_pipeline->profile(zero_rows);
  const bool zero_timing =
      backend == Backend::Cpu
          ? zero_rows[0].timing.available() &&
                zero_rows[0].timing.sample_count == 1u &&
                zero_rows[0].timing.clock == StepClock::HostSteady &&
                zero_rows[0].timing.relation == StepTimingRelation::Exclusive
          : TimingUnavailable(zero_rows[0].timing);
  if (!zero_profile || zero_profile->written != 1u ||
      zero_profile->total != 1u || zero_profile->truncated() ||
      zero_rows[0].index != 0u ||
      zero_rows[0].program != zero_program->fingerprint() ||
      !zero_rows[0].execution.available() ||
      zero_rows[0].execution.sample_count != 1u ||
      zero_rows[0].execution.original_dispatches != 0u ||
      zero_rows[0].execution.final_dispatches != 0u || !zero_timing ||
      zero_rows[0].memory.backend != backend ||
      zero_rows[0].memory.scope != MemoryScope::Pipeline ||
      zero_profile->shared_memory.backend != backend ||
      zero_profile->shared_memory.scope != MemoryScope::Pipeline ||
      zero_profile->execution.backend != backend ||
      zero_profile->execution.dispatches != 0u ||
      zero_profile->execution.command_submits != 0u ||
      zero_profile->execution.pipeline.verified_step_count != 1u ||
      zero_profile->instrumentation_command_count != 0u ||
      zero_profile->instrumentation_byte_count != 0u ||
      zero_profile->referenced_resource_bytes != 0u ||
      !zero_profile->observation.available() ||
      !ProfileMemoryReconciles(*zero_profile, zero_rows)) {
    return 13;
  }
  if (backend != Backend::Cpu) {
    return 0;
  }

  constexpr std::array<std::uint32_t, 4u> valid_indices{0u, 1u, 2u, 3u};
  constexpr std::array<std::uint32_t, 4u> invalid_indices{9u, 1u, 2u, 3u};
  auto gather = on(device)
                    .input<std::int32_t>(input_values.size())
                    .zip_input<std::uint32_t>(valid_indices.size())
                    .branch([](auto values, auto indices) {
                      return values.gather(indices);
                    })
                    .compile();
  auto failure_input = Upload(device, input_values);
  auto failure_middle = device.buffer<std::int32_t>(input_values.size());
  auto failure_indices = Upload(device, valid_indices);
  auto gathered = device.buffer<std::int32_t>(input_values.size());
  auto failure_output = device.buffer<std::int32_t>(input_values.size());
  if (!gather || !failure_input || !failure_middle || !failure_indices ||
      !gathered || !failure_output) {
    return 14;
  }
  auto failure_pipeline =
      pipeline(device)
          .profile(PipelineProfile::Steps)
          .then(*increment, read(*failure_input), write(*failure_middle))
          .then(*gather, read(*failure_middle, *failure_indices),
                write(*gathered))
          .then(*double_value, read(*gathered), write(*failure_output))
          .prepare();
  if (!failure_pipeline || !failure_pipeline->run()) {
    return 15;
  }
  std::array<PipelineStepProfile, 3u> successful_rows{};
  const auto successful_profile = failure_pipeline->profile(successful_rows);
  if (!successful_profile || !successful_rows[0].execution.available() ||
      !successful_rows[1].execution.available() ||
      !successful_rows[2].execution.available() ||
      !successful_rows[0].timing.available() ||
      !successful_rows[1].timing.available() ||
      !successful_rows[2].timing.available() ||
      !Overwrite(*failure_indices, invalid_indices)) {
    return 16;
  }
  const Status failure = failure_pipeline->run();
  std::array<PipelineStepProfile, 3u> failure_rows{};
  const auto failure_profile = failure_pipeline->profile(failure_rows);
  if (failure || failure.reason() != Reason::GatherIndexOutOfRange ||
      !failure_profile || failure_profile->written != 3u ||
      failure_profile->total != 3u || failure_profile->truncated() ||
      failure_profile->execution.pipeline.verified_step_count != 1u ||
      failure_profile->execution.pipeline.failed_step_index != 1u ||
      failure_rows[0].index != 0u || failure_rows[1].index != 1u ||
      failure_rows[2].index != 2u ||
      failure_rows[0].program != increment->fingerprint() ||
      failure_rows[1].program != gather->fingerprint() ||
      failure_rows[2].program != double_value->fingerprint() ||
      !failure_rows[0].execution.available() ||
      !failure_rows[1].execution.available() ||
      failure_rows[2].execution.available() ||
      !failure_rows[0].timing.available() ||
      !failure_rows[1].timing.available() ||
      failure_rows[2].timing.available() ||
      failure_profile->referenced_resource_bytes != sizeof(input_values) * 5u ||
      !ProfileMemoryReconciles(*failure_profile, failure_rows)) {
    return 17;
  }
  return 0;
}

} // namespace rund_node_test_pipeline
