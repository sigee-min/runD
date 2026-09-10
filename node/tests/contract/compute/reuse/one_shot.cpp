#include "local.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <tuple>
#include <vector>

namespace rund_node_test_compute_reuse {
namespace {

template <class Program>
[[nodiscard]] std::uint64_t OneShotPhysicalBytes(const Program &program) {
  using namespace rund::compute;
  std::uint64_t bytes = program.graph().memory.physical_bytes;
  for (const graph::Resource &resource : program.graph().resources) {
    if (resource.visibility != graph::Visibility::Internal) {
      bytes += resource.bytes;
    }
  }
  return bytes;
}

template <class Job>
[[nodiscard]] std::uint64_t PendingInputBytes(const Job &job) {
  using namespace rund::compute;
  std::array<MemoryEntry, 64u> entries{};
  const MemorySnapshot snapshot = job.memory_snapshot(entries);
  if (snapshot.truncated()) {
    return 0u;
  }
  std::uint64_t bytes = 0u;
  for (std::size_t index = 0u; index < snapshot.written; ++index) {
    const MemoryEntry &entry = entries[index];
    if (entry.use == MemoryUse::PendingInput &&
        (entry.category == MemoryCategory::Host ||
         entry.category == MemoryCategory::Device)) {
      bytes += entry.bytes.current;
    }
  }
  return bytes;
}

} // namespace

[[nodiscard]] int CheckReadOnlyOneShot(rund::compute::Device &device) {
  using namespace rund::compute;
  const std::array<std::int32_t, 4u> input{1, 2, 3, 4};

  auto bounded = on(device)
                     .map<std::int32_t>("one-shot-bounded", input.size(),
                                        [](auto value) { return value; })
                     .filter([](auto value) { return value > 2; })
                     .compile();
  auto outputs = on(device)
                     .map<std::int32_t>("one-shot-outputs", input.size(),
                                        [](auto value) { return value * 2; })
                     .branch([](auto doubled) {
                       auto plus_one =
                           doubled.map("one-shot-plus-one",
                                       [](auto value) { return value + 1; });
                       return rund::compute::outputs(doubled, plus_one);
                     })
                     .compile();
  if (!bounded || !outputs) {
    return 1;
  }

  const std::uint64_t bounded_bytes = OneShotPhysicalBytes(*bounded);
  const MemoryStats bounded_before = device.memory();
  auto bounded_result = bounded->run(input);
  const MemoryStats bounded_after = device.memory();
  if (!bounded_result || *bounded_result != std::vector<std::int32_t>{3, 4} ||
      bounded_bytes == 0u ||
      bounded_after.host.current != bounded_before.host.current ||
      bounded_after.host.cumulative - bounded_before.host.cumulative !=
          bounded_bytes) {
    std::fprintf(
        stderr, "bounded one-shot physical=%llu expected=%llu\n",
        static_cast<unsigned long long>(bounded_after.host.cumulative -
                                        bounded_before.host.cumulative),
        static_cast<unsigned long long>(bounded_bytes));
    return 2;
  }

  const std::uint64_t output_bytes = OneShotPhysicalBytes(*outputs);
  const MemoryStats output_before = device.memory();
  auto output_result = outputs->run(input);
  const MemoryStats output_after = device.memory();
  if (!output_result ||
      std::get<0>(*output_result) != std::vector<std::int32_t>{2, 4, 6, 8} ||
      std::get<1>(*output_result) != std::vector<std::int32_t>{3, 5, 7, 9} ||
      output_bytes == 0u ||
      output_after.host.current != output_before.host.current ||
      output_after.host.cumulative - output_before.host.cumulative !=
          output_bytes) {
    std::fprintf(stderr, "multi-output one-shot physical=%llu expected=%llu\n",
                 static_cast<unsigned long long>(output_after.host.cumulative -
                                                 output_before.host.cumulative),
                 static_cast<unsigned long long>(output_bytes));
    return 3;
  }

  const MemoryStats resident_before = device.memory();
  auto resident = outputs->resident(input);
  if (!resident ||
      PendingInputBytes(*resident) != input.size() * sizeof(input.front())) {
    return 4;
  }
  const MemoryStats resident_after = device.memory();
  if (resident_after.host.cumulative - resident_before.host.cumulative !=
          output_bytes + input.size() * sizeof(input.front()) ||
      !resident->run()) {
    return 5;
  }
  const std::array<std::int32_t, 4u> next{5, 6, 7, 8};
  if (!resident->write(next) || !resident->run()) {
    return 6;
  }
  auto next_output = resident->read_all();
  if (!next_output ||
      std::get<0>(*next_output) != std::vector<std::int32_t>{10, 12, 14, 16} ||
      std::get<1>(*next_output) != std::vector<std::int32_t>{11, 13, 15, 17}) {
    return 7;
  }

  auto identity = device.info();
  const Stats execution = resident->stats();
  const MemoryStats memory = resident->memory();
  const auto profile = resident->profile();
  if (!identity || !profile || profile->device() != *identity ||
      profile->execution().graph_hash != execution.graph_hash ||
      profile->execution().output_hash != execution.output_hash ||
      profile->execution().dispatches != execution.dispatches ||
      !SameMemory(profile->memory(), memory) ||
      profile->memory().scope != MemoryScope::Job ||
      profile->kernel_time().numerator != execution.kernel_ns ||
      profile->kernel_time().denominator != execution.kernel_samples ||
      profile->dispatches_per_submit().numerator != execution.dispatches ||
      profile->command_pressure().numerator !=
          execution.command_inflight_peak ||
      profile->command_pressure().denominator != execution.command_capacity ||
      profile->pipeline_cache().selected != execution.pipeline_cache_hits ||
      profile->buffer_reuse().selected != execution.buffer_reuses ||
      profile->descriptor_reuse().selected != execution.descriptor_reuses ||
      profile->dispatch_reduction().denominator !=
          execution.original_dispatches ||
      profile->internal_traffic().selected !=
          execution.internal_roundtrip_bytes) {
    return 8;
  }
  constexpr MemoryCategory categories[]{
      MemoryCategory::Host,     MemoryCategory::Frame,   MemoryCategory::Tile,
      MemoryCategory::Resident, MemoryCategory::Staging, MemoryCategory::Device,
      MemoryCategory::Transfer,
  };
  for (const MemoryCategory category : categories) {
    const auto usage = profile->memory_usage(category);
    if (!usage ||
        (category == MemoryCategory::Transfer && usage->available())) {
      return 9;
    }
  }
  const auto invalid =
      profile->memory_usage(static_cast<MemoryCategory>(0xffu));
  if (invalid || invalid.code() != Code::Invalid ||
      invalid.error() != "compute_profile_memory_category_invalid") {
    return 10;
  }

  auto retained = std::move(*resident);
  const auto moved = resident->profile();
  if (moved || moved.code() != Code::Invalid ||
      moved.error() != "compute_profile_invalid" || !retained.profile()) {
    return 11;
  }
  return 0;
}

} // namespace rund_node_test_compute_reuse
