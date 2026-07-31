#include <rund/compute.hpp>

#include "allocation.hpp"
#include "src/compute/memory/profile.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

namespace {

using TelemetryProfile = rund::compute::telemetry::Profile;
using TelemetryRate = rund::compute::telemetry::Rate;
using TelemetryShare = rund::compute::telemetry::Share;

constexpr TelemetryRate kMissingRate{};
constexpr TelemetryRate kExactRate{.numerator = 14u, .denominator = 4u};
constexpr TelemetryRate kSaturatedRate{
    .numerator = std::numeric_limits<std::uint64_t>::max(), .denominator = 4u};
constexpr TelemetryShare kMissingShare{};
constexpr TelemetryShare kWideShare{
    .selected = std::numeric_limits<std::uint64_t>::max() - 1u,
    .other = std::numeric_limits<std::uint64_t>::max() - 1u};

static_assert(!kMissingRate.available() && !kMissingRate.value().has_value());
static_assert(kExactRate.available() && kExactRate.value() == 3.5L);
static_assert(kSaturatedRate.saturated() && !kSaturatedRate.available() &&
              !kSaturatedRate.value().has_value());
static_assert(!kMissingShare.available() && !kMissingShare.value().has_value());
static_assert(kWideShare.available() && kWideShare.value() == 0.5L);
static_assert(!std::is_default_constructible_v<TelemetryProfile>);
static_assert(!std::is_aggregate_v<TelemetryProfile>);
static_assert(std::is_nothrow_move_constructible_v<TelemetryProfile>);

[[nodiscard]] bool CheckTelemetryMath() {
  using namespace rund::compute;
  using namespace rund::compute::telemetry;

  const Stats zero{.command_capacity = 8u,
                   .command_inflight_peak = 5u,
                   .original_dispatches = 9u,
                   .final_dispatches = 4u,
                   .kernel_samples = 1u};
  const Profile zero_profile = detail::ProfileAccess::make(
      DeviceInfo{.name = "test", .driver = "test", .driver_details = ""}, zero,
      {});
  const Focus zero_focus = zero_profile.largest_time();
  if (!zero_focus.available() || zero_focus.nanoseconds() != 0u ||
      zero_focus.saturated() || !zero_focus.includes(Stage::Kernel) ||
      zero_focus.includes(Stage::ShaderCompile) ||
      zero_profile.dispatch_reduction() != Rate{5u, 9u} ||
      zero_profile.command_pressure() != Rate{5u, 8u}) {
    return false;
  }

  constexpr std::uint64_t limit = std::numeric_limits<std::uint64_t>::max();
  const Stats tied{.kernel_ns = limit,
                   .kernel_samples = 1u,
                   .shader_compile_ns = limit,
                   .descriptor_setup_ns = 17u,
                   .readback_ns = limit};
  const Profile tied_profile = detail::ProfileAccess::make(
      DeviceInfo{.name = "test", .driver = "test", .driver_details = ""}, tied,
      {});
  const Focus tied_focus = tied_profile.largest_time();
  return tied_focus.available() && tied_focus.saturated() &&
         tied_focus.nanoseconds() == limit &&
         tied_focus.includes(Stage::ShaderCompile) &&
         tied_focus.includes(Stage::Kernel) &&
         tied_focus.includes(Stage::Readback) &&
         !tied_focus.includes(Stage::SpirvCompile) &&
         !tied_focus.includes(Stage::PipelineCreate) &&
         !tied_focus.includes(Stage::DescriptorSetup) &&
         !tied_focus.includes(Stage::SubmitWait);
}

[[nodiscard]] bool
SameMemory(const rund::compute::MemoryStats &left,
           const rund::compute::MemoryStats &right) noexcept {
  return left.backend == right.backend && left.scope == right.scope &&
         left.host.current == right.host.current &&
         left.frame.current == right.frame.current &&
         left.tile.current == right.tile.current &&
         left.resident.current == right.resident.current &&
         left.staging.current == right.staging.current &&
         left.device.current == right.device.current &&
         left.transfer.current == right.transfer.current;
}

[[nodiscard]] std::uint64_t HashBytes(const void *const data,
                                      const std::size_t bytes) noexcept {
  constexpr std::uint64_t offset = 1469598103934665603ull;
  constexpr std::uint64_t prime = 1099511628211ull;
  const auto *const values = static_cast<const std::uint8_t *>(data);
  std::uint64_t hash = offset;
  for (std::size_t index = 0u; index < bytes; ++index) {
    hash ^= values[index];
    hash *= prime;
  }
  return hash;
}

static_assert(sizeof(rund::compute::Run) == 1024u);
static_assert(alignof(rund::compute::Run) == alignof(std::uint64_t));
static_assert(std::is_nothrow_copy_constructible_v<rund::compute::Run>);
static_assert(std::is_nothrow_move_constructible_v<rund::compute::Run>);

template <class T, std::size_t Count>
[[nodiscard]] bool CheckHostIdentity(rund::compute::Device &device,
                                     const std::string_view name,
                                     const std::array<T, Count> &input) {
  auto program =
      rund::compute::on(device)
          .template map<T>(name, Count, [](auto value) { return value; })
          .compile();
  if (!program) {
    std::fprintf(stderr, "host identity compile failed name=%.*s reason=%.*s\n",
                 static_cast<int>(name.size()), name.data(),
                 static_cast<int>(program.error().size()),
                 program.error().data());
    return false;
  }
  auto output = program->run(std::span<const T>{input});
  const std::vector<T> expected{input.begin(), input.end()};
  const bool aligned =
      !output || output->empty() ||
      reinterpret_cast<std::uintptr_t>(output->data()) % alignof(T) == 0u;
  if (!output || *output != expected || !aligned) {
    std::fprintf(
        stderr, "host identity failed name=%.*s read=%d count=%zu aligned=%d\n",
        static_cast<int>(name.size()), name.data(), output ? 1 : 0,
        output ? output->size() : 0u, aligned ? 1 : 0);
    return false;
  }
  return true;
}

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
    if (!profile->memory_usage(category)) {
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

[[nodiscard]] bool CheckProgramConcurrency(rund::compute::Device &device) {
  using namespace rund::compute;
  constexpr std::size_t count = 32u;
  std::array<std::int32_t, count> first{};
  std::array<std::int32_t, count> second{};
  std::array<std::int32_t, count> first_expected{};
  std::array<std::int32_t, count> second_expected{};
  for (std::size_t index = 0u; index < count; ++index) {
    first[index] = static_cast<std::int32_t>(index);
    second[index] = -static_cast<std::int32_t>(index);
    first_expected[index] = first[index] * 3 + 1;
    second_expected[index] = second[index] * 3 + 1;
  }

  auto compiled =
      on(device)
          .map<std::int32_t>("program-concurrency", count,
                             [](auto value) { return value * 3 + 1; })
          .compile();
  if (!compiled) {
    return false;
  }
  auto program = std::move(compiled).value();
  std::atomic<bool> failed{false};
  const auto run = [&](const std::array<std::int32_t, count> &input,
                       const std::array<std::int32_t, count> &expected) {
    for (std::size_t iteration = 0u; iteration < 128u; ++iteration) {
      auto output = program.run(std::span<const std::int32_t>{input});
      if (!output || output->size() != expected.size() ||
          !std::equal(output->begin(), output->end(), expected.begin())) {
        failed.store(true, std::memory_order_relaxed);
        return;
      }
    }
  };
  std::thread first_run{run, std::cref(first), std::cref(first_expected)};
  std::thread second_run{run, std::cref(second), std::cref(second_expected)};
  std::thread observer{[&] {
    for (std::size_t iteration = 0u; iteration < 128u; ++iteration) {
      if (program.memory().scope != MemoryScope::Program) {
        failed.store(true, std::memory_order_relaxed);
        return;
      }
    }
  }};
  first_run.join();
  second_run.join();
  observer.join();
  return !failed.load(std::memory_order_relaxed);
}

[[nodiscard]] bool CheckProgramLifetime(rund::compute::Device &device) {
  using namespace rund::compute;
  const std::array<std::int32_t, 4u> input{1, 2, 3, 4};

  const MemoryStats before = device.memory();
  {
    auto compiled =
        on(device)
            .map<std::int32_t>("program-cache-lifetime", input.size(),
                               [](auto value) { return value + 1; })
            .compile();
    if (!compiled || !compiled->run(input)) {
      return false;
    }
  }
  if (!SameMemory(before, device.memory())) {
    std::fprintf(stderr, "program convenience cache outlived its owner\n");
    return false;
  }

  auto source = device.upload(std::span<const std::int32_t>{input});
  auto target = device.buffer<std::int32_t>(input.size());
  if (!source || !target) {
    return false;
  }
  std::optional<Run> retained;
  {
    auto compiled = on(device)
                        .map<std::int32_t>("run-program-lifetime", input.size(),
                                           [](auto value) { return value + 1; })
                        .compile();
    if (!compiled) {
      return false;
    }
    auto run = compiled->run(*source, *target);
    if (!run) {
      return false;
    }
    retained.emplace(std::move(run).value());
  }
  std::array<std::int32_t, 4u> output{};
  return retained->read(*target, std::span<std::int32_t>{output}) &&
         output == std::array<std::int32_t, 4u>{2, 3, 4, 5};
}

} // namespace

int RunComputeReuseContract() {
  if (!CheckTelemetryMath()) {
    return 1;
  }
  auto device = rund::compute::open(rund::compute::Target::cpu());
  if (!device) {
    return 2;
  }
  auto program = rund::compute::on(device.value())
                     .map<std::int32_t>(
                         "reuse", 4, [](auto value) { return value * 2 + 5; })
                     .compile();
  if (!program) {
    return 2;
  }

  const std::array<std::int32_t, 4> input{1, 2, 3, 4};
  {
    auto warm_host = program.value().run(std::span<const std::int32_t>{input});
    if (!warm_host) {
      return 10;
    }
  }
  node_compute_allocation::Start();
  auto host = program.value().run(std::span<const std::int32_t>{input});
  node_compute_allocation::Stop();
  const std::uint64_t host_allocations = node_compute_allocation::Count();
  if (!host || host_allocations != 1u ||
      *host != std::vector<std::int32_t>{7, 9, 11, 13}) {
    std::fprintf(stderr, "host one-shot heap allocations=%llu\n",
                 static_cast<unsigned long long>(host_allocations));
    return 10;
  }
  if (!CheckProgramConcurrency(device.value())) {
    std::fprintf(stderr, "program convenience execution is not isolated\n");
    return 12;
  }
  if (!CheckProgramLifetime(device.value())) {
    std::fprintf(stderr, "program/run ownership contract failed\n");
    return 13;
  }

  using Q16_16 = rund::compute::Fixed<16u, 16u>;
  using Q20_44 = rund::compute::Fixed<20u, 44u>;
  static_assert(sizeof(Q16_16) == sizeof(std::uint32_t));
  static_assert(sizeof(Q20_44) == sizeof(std::uint64_t));
  if (!CheckHostIdentity(device.value(), "host-i32",
                         std::array<std::int32_t, 3>{-7, 0, 11}) ||
      !CheckHostIdentity(device.value(), "host-u32",
                         std::array<std::uint32_t, 3>{0u, 7u, 11u}) ||
      !CheckHostIdentity(device.value(), "host-i64",
                         std::array<std::int64_t, 3>{-7, 0, 11}) ||
      !CheckHostIdentity(device.value(), "host-u64",
                         std::array<std::uint64_t, 3>{0u, 7u, 11u}) ||
      !CheckHostIdentity(device.value(), "host-q16-16",
                         std::array<Q16_16, 3>{Q16_16::from_raw(-7),
                                               Q16_16::zero(),
                                               Q16_16::from_raw(11)}) ||
      !CheckHostIdentity(device.value(), "host-q20-44",
                         std::array<Q20_44, 3>{Q20_44::from_raw(-7),
                                               Q20_44::zero(),
                                               Q20_44::from_raw(11)}) ||
      !CheckHostIdentity(device.value(), "host-empty",
                         std::array<std::int32_t, 0>{})) {
    return 11;
  }
  if (const int one_shot = CheckReadOnlyOneShot(device.value());
      one_shot != 0) {
    return 20 + one_shot;
  }
  auto source = device.value().upload(std::span<const std::int32_t>{input});
  auto target = device.value().buffer<std::int32_t>(input.size());
  if (!source || !target) {
    return 3;
  }

  auto first = program.value().run(source.value(), target.value());
  node_compute_allocation::Start();
  auto second = program.value().run(source.value(), target.value());
  node_compute_allocation::Stop();
  if (!first || !second) {
    return 4;
  }
  if (node_compute_allocation::Count() != 0u) {
    std::fprintf(
        stderr, "reuse warm heap allocations=%llu\n",
        static_cast<unsigned long long>(node_compute_allocation::Count()));
    return 5;
  }

  node_compute_allocation::Start();
  rund::compute::Run copied = second.value();
  rund::compute::Run moved = std::move(copied);
  node_compute_allocation::Stop();
  if (node_compute_allocation::Count() != 0u ||
      moved.stats().graph_hash != second.value().stats().graph_hash) {
    std::fprintf(
        stderr, "run receipt copy heap allocations=%llu\n",
        static_cast<unsigned long long>(node_compute_allocation::Count()));
    return 5;
  }
  std::array<std::int32_t, 4> copied_output{};
  if (!moved.read(target.value(), std::span<std::int32_t>{copied_output}) ||
      moved.stats().download_events != 1u ||
      second.value().stats().download_events != 0u ||
      copied_output != std::array<std::int32_t, 4>{7, 9, 11, 13}) {
    std::fprintf(stderr, "run receipt telemetry copy did not diverge\n");
    return 5;
  }

  const rund::compute::Stats warm = second.value().stats();
  if (warm.backend != rund::compute::Backend::Cpu ||
      warm.pipeline_compiles != 0 || warm.buffer_allocations != 0 ||
      warm.download_events != 0) {
    std::fprintf(stderr,
                 "warm stats pipeline_compile=%llu buffer_allocation=%llu "
                 "download_event=%llu\n",
                 static_cast<unsigned long long>(warm.pipeline_compiles),
                 static_cast<unsigned long long>(warm.buffer_allocations),
                 static_cast<unsigned long long>(warm.download_events));
    return 6;
  }
  if (warm.dispatches != 1 || warm.graph_hash == 0 || warm.output_hash != 0) {
    std::fprintf(stderr, "warm identity submit=%llu graph=%llu output=%llu\n",
                 static_cast<unsigned long long>(warm.dispatches),
                 static_cast<unsigned long long>(warm.graph_hash),
                 static_cast<unsigned long long>(warm.output_hash));
    return 7;
  }

  std::array<std::int32_t, 4> output{};
  if (!second.value().read(target.value(), std::span<std::int32_t>{output})) {
    return 8;
  }
  const rund::compute::Stats read = second.value().stats();
  if (read.download_events != 1 || read.downloaded_bytes != sizeof(output) ||
      read.output_hash != HashBytes(output.data(), sizeof(output)) ||
      output != std::array<std::int32_t, 4>{7, 9, 11, 13}) {
    std::fprintf(stderr,
                 "read stats download_events=%llu downloaded_bytes=%llu "
                 "output=%llu\n",
                 static_cast<unsigned long long>(read.download_events),
                 static_cast<unsigned long long>(read.downloaded_bytes),
                 static_cast<unsigned long long>(read.output_hash));
    return 9;
  }
  return 0;
}
