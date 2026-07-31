#pragma once

#include "api.hpp"

#include <rund/compute.hpp>
#include <rund/compute/math.hpp>
#include <rund/compute/session.hpp>
#include <rund/rund.hpp>

#if defined(RUND_COMPUTE_FOCUS)
#include <kernel/program/compute/matrix/tile.hpp>
#include <kernel/program/compute/transform/stage.hpp>
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace rund::measure::compute {

#if defined(RUND_COMPUTE_FOCUS)
[[nodiscard]] inline bool ParseBackend(const std::string_view name,
                                       Backend &backend) noexcept {
  for (const Backend candidate : kBackends) {
    if (name == Name(candidate)) {
      backend = candidate;
      return true;
    }
  }
  return false;
}
#endif

inline void PrintCsv(const std::string_view text) {
  std::putchar('"');
  for (const char value : text) {
    if (value == '"') {
      std::fputs("\"\"", stdout);
    } else {
      std::putchar(static_cast<unsigned char>(value));
    }
  }
  std::putchar('"');
}

inline bool ReportEnvironment(const Backend backend) {
  auto device = rund::compute::open(TargetFor(backend));
  if (!device) {
    std::printf("environment,%s,open_failed,%u,", Name(backend),
                static_cast<unsigned>(device.code()));
    PrintCsv(device.error());
    std::fputs(",\"\",\"\",\"\"\n", stdout);
    return false;
  }
  auto info = device->info();
  if (!info) {
    std::printf("environment,%s,info_failed,%u,", Name(backend),
                static_cast<unsigned>(info.code()));
    PrintCsv(info.error());
    std::fputs(",\"\",\"\",\"\"\n", stdout);
    return false;
  }
  if (info->backend != backend) {
    std::printf("environment,%s,backend_mismatch,%u,", Name(backend),
                static_cast<unsigned>(rund::compute::Code::Invalid));
    PrintCsv("compute_device_info_backend_mismatch");
    std::putchar(',');
    PrintCsv(info->name);
    std::putchar(',');
    PrintCsv(info->driver);
    std::putchar(',');
    PrintCsv(info->driver_details);
    std::putchar('\n');
    return false;
  }
  std::printf("environment,%s,ok,%u,\"\",", Name(backend),
              static_cast<unsigned>(rund::compute::Code::Ok));
  PrintCsv(info->name);
  std::putchar(',');
  PrintCsv(info->driver);
  std::putchar(',');
  PrintCsv(info->driver_details);
  std::putchar('\n');
  return true;
}

struct WarmCounterSample final {
  std::uint64_t pipeline_compiles;
  std::uint64_t buffer_allocations;
  std::uint64_t descriptor_pool_creations;
  std::uint64_t descriptor_set_allocations;
  std::uint64_t download_events;
  std::uint64_t uploaded_bytes;
  std::uint64_t downloaded_bytes;
  std::uint64_t internal_roundtrip_bytes;
  std::uint64_t external_roundtrip_bytes;
};

consteval bool WarmCounterContract() {
  WarmCounters counters{};
  if (!counters.zero()) {
    return false;
  }
  counters.observe(WarmCounterSample{1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u});
  counters.observe(WarmCounterSample{9u, 8u, 7u, 6u, 5u, 4u, 3u, 2u, 1u});
  if (counters.pipeline_compiles != 10u ||
      counters.buffer_allocations != 10u ||
      counters.descriptor_pool_creations != 10u ||
      counters.descriptor_set_allocations != 10u ||
      counters.download_events != 10u || counters.uploaded_bytes != 10u ||
      counters.downloaded_bytes != 10u ||
      counters.internal_roundtrip_bytes != 10u ||
      counters.external_roundtrip_bytes != 10u ||
      counters.zero()) {
    return false;
  }
  constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
  counters.observe(WarmCounterSample{maximum, maximum, maximum, maximum,
                                     maximum, maximum, maximum, maximum,
                                     maximum});
  return counters.pipeline_compiles == maximum &&
         counters.buffer_allocations == maximum &&
         counters.descriptor_pool_creations == maximum &&
         counters.descriptor_set_allocations == maximum &&
         counters.download_events == maximum &&
         counters.uploaded_bytes == maximum &&
         counters.downloaded_bytes == maximum &&
         counters.internal_roundtrip_bytes == maximum &&
         counters.external_roundtrip_bytes == maximum;
}

static_assert(WarmCounterContract());

#if defined(RUND_COMPUTE_FOCUS)
struct BulkCost final {
  std::uint64_t count;
  std::uint64_t passes;
  std::uint64_t logical_ops;
  std::uint64_t value_bytes;
  std::uint64_t coefficient_bytes;

  [[nodiscard]] constexpr std::uint64_t logical_bytes() const noexcept {
    return value_bytes + coefficient_bytes;
  }
};

consteval BulkCost MatrixCost(const std::uint64_t side) {
  constexpr std::uint64_t tile = rund::kernel::matrix_tile::Side;
  const std::uint64_t tiles = (side + tile - 1u) / tile;
  const std::uint64_t elements = side * side;
  const std::uint64_t products = elements * side;
  const std::uint64_t tile_values = tile * tile;
  const std::uint64_t loaded_values = tiles * tiles * tiles * 2u * tile_values;
  return BulkCost{
      .count = elements,
      .passes = tiles,
      .logical_ops = products * 2u,
      .value_bytes = (loaded_values + elements) * sizeof(std::int32_t),
      .coefficient_bytes = 0u,
  };
}

consteval BulkCost TransformCost(const std::uint64_t count) {
  const std::uint64_t stages = std::countr_zero(count);
  const std::uint64_t passes = rund::kernel::transform_stage::Dispatches(count);
  return BulkCost{
      .count = count,
      .passes = passes,
      .logical_ops = stages * (count / 2u) * 10u,
      .value_bytes = count * passes * sizeof(Fixed<1, 31>) * 4u,
      .coefficient_bytes = stages * (count / 2u) * sizeof(Fixed<1, 31>) * 2u,
  };
}

constexpr BulkCost MatrixBulkCost = MatrixCost(512u);
constexpr BulkCost TransformBulkCost = TransformCost(1u << 20u);
static_assert(MatrixBulkCost.logical_ops == 268'435'456u);
static_assert(MatrixBulkCost.value_bytes == 34'603'008u);
static_assert(MatrixBulkCost.coefficient_bytes == 0u);
static_assert(MatrixBulkCost.logical_bytes() == 34'603'008u);
static_assert(TransformBulkCost.passes == 7u);
static_assert(TransformBulkCost.logical_ops == 104'857'600u);
static_assert(TransformBulkCost.value_bytes == 117'440'512u);
static_assert(TransformBulkCost.coefficient_bytes == 83'886'080u);
static_assert(TransformBulkCost.logical_bytes() == 201'326'592u);
#endif

[[nodiscard]] constexpr bool
ValidInflight(const std::uint64_t expected_capacity,
              const std::uint64_t observed_capacity,
              const std::uint64_t observed_peak,
              const std::uint64_t rejections) noexcept {
  return expected_capacity != 0u && observed_capacity == expected_capacity &&
         observed_peak != 0u && observed_peak <= observed_capacity &&
         rejections == 0u;
}

static_assert(ValidInflight(8u, 8u, 1u, 0u));
static_assert(ValidInflight(8u, 8u, 8u, 0u));
static_assert(!ValidInflight(8u, 8u, 0u, 0u));
static_assert(!ValidInflight(8u, 8u, 9u, 0u));
static_assert(!ValidInflight(8u, 7u, 5u, 0u));
static_assert(!ValidInflight(8u, 8u, 5u, 1u));

#if !defined(RUND_COMPUTE_FOCUS)
inline void AccumulateInflight(rund::compute::Stats &total,
                               const rund::compute::Stats &job) noexcept {
  total.backend = job.backend;
  ::rund::detail::counter::Accumulate(total.command_submits,
                                      job.command_submits);
  ::rund::detail::counter::Accumulate(total.dispatches, job.dispatches);
  ::rund::detail::counter::Accumulate(total.command_capacity_rejections,
                                      job.command_capacity_rejections);
  total.command_capacity =
      std::max(total.command_capacity, job.command_capacity);
  total.command_inflight_peak =
      std::max(total.command_inflight_peak, job.command_inflight_peak);
}
#endif

inline void PrintStatsColumns() {
  std::fputs(
      ",pipeline_compiles,buffer_allocations,download_events,dispatches,"
      "command_submits,uploaded_bytes,downloaded_bytes,pipeline_cache_hits,"
      "pipeline_cache_evictions,buffer_reuses,descriptor_pool_creations,"
      "descriptor_set_allocations,descriptor_reuses,original_dispatches,"
      "final_dispatches,fusions,fusion_rejections,internal_roundtrip_bytes,"
      "external_roundtrip_bytes,kernel_ns,kernel_samples,shader_compile_ns,"
      "spirv_compile_ns,pipeline_create_ns,descriptor_setup_ns,submit_wait_ns,"
      "readback_ns,graph_hash,output_hash,workers,participating,tiles,tile_"
      "size,"
      "vector_chunks,tail_chunks",
      stdout);
}

template <class Stats> void PrintStats(const Stats &stats) {
  const auto field = [](const std::uint64_t value) {
    std::printf(",%llu", static_cast<unsigned long long>(value));
  };
  field(stats.pipeline_compiles);
  field(stats.buffer_allocations);
  field(stats.download_events);
  field(stats.dispatches);
  field(stats.command_submits);
  field(stats.uploaded_bytes);
  field(stats.downloaded_bytes);
  field(stats.pipeline_cache_hits);
  field(stats.pipeline_cache_evictions);
  field(stats.buffer_reuses);
  field(stats.descriptor_pool_creations);
  field(stats.descriptor_set_allocations);
  field(stats.descriptor_reuses);
  field(stats.original_dispatches);
  field(stats.final_dispatches);
  field(stats.fusions);
  field(stats.fusion_rejections);
  field(stats.internal_roundtrip_bytes);
  field(stats.external_roundtrip_bytes);
  field(stats.kernel_ns);
  field(stats.kernel_samples);
  field(stats.shader_compile_ns);
  field(stats.spirv_compile_ns);
  field(stats.pipeline_create_ns);
  field(stats.descriptor_setup_ns);
  field(stats.submit_wait_ns);
  field(stats.readback_ns);
  field(stats.graph_hash);
  field(stats.output_hash);
  field(stats.worker_count);
  field(stats.participating_workers);
  field(stats.tile_count);
  field(stats.tile_size);
  field(stats.vector_chunks);
  field(stats.tail_chunks);
}

inline void PrintWarmColumns() {
  std::fputs(",warm_pipeline_compiles,warm_buffer_allocations,"
             "warm_download_events,warm_uploaded_bytes,warm_zero",
             stdout);
}

inline void PrintWorkloadColumns() {
  std::fputs(
      "workload_columns,backend,family,variant,status,input_count,active_count,"
      "samples,median_us,input_items_per_s,active_items_per_s",
      stdout);
  PrintStatsColumns();
  PrintWarmColumns();
  std::fputs(",resident_bytes,staging_bytes\n", stdout);
}

inline void PrintWarm(const WarmCounters &warm) {
  std::printf(",%llu,%llu,%llu,%llu,%u",
              static_cast<unsigned long long>(warm.pipeline_compiles),
              static_cast<unsigned long long>(warm.buffer_allocations),
              static_cast<unsigned long long>(warm.download_events),
              static_cast<unsigned long long>(warm.uploaded_bytes),
              warm.zero() ? 1u : 0u);
}

struct HashEvidence final {
  std::uint64_t graph = 0u;
  std::uint64_t output = 0u;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return graph != 0u && output != 0u;
  }

  [[nodiscard]] constexpr bool
  operator==(const HashEvidence &) const noexcept = default;
};

struct ReferenceKey final {
  std::string_view workload{};
  std::string_view variant{};
  std::size_t count = 0u;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return !workload.empty() && !variant.empty();
  }

  [[nodiscard]] constexpr bool
  operator==(const ReferenceKey &) const noexcept = default;
};

enum class ReferenceStatus : std::uint8_t {
  Established,
  Matched,
  Missing,
  Mismatch,
  Full,
  Invalid,
};

template <std::size_t Capacity> struct ReferenceLedger final {
  struct Entry final {
    ReferenceKey key{};
    HashEvidence evidence{};
  };

  std::array<Entry, Capacity> entries{};
  std::size_t size = 0u;

  [[nodiscard]] constexpr ReferenceStatus
  check(const bool establish, const ReferenceKey key,
        const HashEvidence evidence) noexcept {
    if (!key.valid() || !evidence.valid()) {
      return ReferenceStatus::Invalid;
    }
    for (std::size_t index = 0u; index < size; ++index) {
      if (entries[index].key == key) {
        return entries[index].evidence == evidence ? ReferenceStatus::Matched
                                                   : ReferenceStatus::Mismatch;
      }
    }
    if (!establish) {
      return ReferenceStatus::Missing;
    }
    if (size == Capacity) {
      return ReferenceStatus::Full;
    }
    entries[size++] = Entry{.key = key, .evidence = evidence};
    return ReferenceStatus::Established;
  }
};

consteval bool ReferenceLedgerContract() {
  ReferenceLedger<2u> ledger{};
  constexpr ReferenceKey first{"family", "map", 4096u};
  constexpr ReferenceKey second{"collective", "scan", 262144u};
  constexpr ReferenceKey third{"fixed", "wide", 4096u};
  constexpr HashEvidence evidence{11u, 13u};
  return ledger.check(true, first, evidence) == ReferenceStatus::Established &&
         ledger.check(false, first, evidence) == ReferenceStatus::Matched &&
         ledger.check(false, first, HashEvidence{11u, 17u}) ==
             ReferenceStatus::Mismatch &&
         ledger.check(false, second, evidence) == ReferenceStatus::Missing &&
         ledger.check(true, second, evidence) == ReferenceStatus::Established &&
         ledger.check(true, third, evidence) == ReferenceStatus::Full &&
         ledger.size == 2u;
}

static_assert(ReferenceLedgerContract());

constexpr std::size_t ReferenceCapacity = 64u;
constexpr std::size_t RequiredReferenceEntries = 37u;
static_assert(ReferenceCapacity >= RequiredReferenceEntries);
inline ReferenceLedger<ReferenceCapacity> references{};
inline HashEvidence batch_reference{};

template <class Job, class Validate>
bool CaptureOutput(Job &job, const Backend backend,
                   const std::string_view workload, HashEvidence &evidence,
                   Validate validate) {
  const bool valid = [&] {
    if constexpr (requires { job.read_all(); }) {
      const auto values = job.read_all();
      return values && validate(*values);
    } else {
      const auto values = job.read();
      return values && validate(*values);
    }
  }();
  if (!valid) {
    std::fprintf(stderr, "output %s/%.*s validation failed\n", Name(backend),
                 static_cast<int>(workload.size()), workload.data());
    return false;
  }
  const auto stats = job.stats();
  evidence =
      HashEvidence{.graph = stats.graph_hash, .output = stats.output_hash};
  if (!evidence.valid()) {
    std::fprintf(
        stderr, "output %s/%.*s hash missing: graph=%llu output=%llu\n",
        Name(backend), static_cast<int>(workload.size()), workload.data(),
        static_cast<unsigned long long>(evidence.graph),
        static_cast<unsigned long long>(evidence.output));
    return false;
  }
  return true;
}

template <class Job>
bool CaptureOutput(Job &job, const Backend backend,
                   const std::string_view workload, HashEvidence &evidence) {
  return CaptureOutput(job, backend, workload, evidence,
                       [](const auto &) { return true; });
}

inline bool CheckReference(const Backend backend, const ReferenceKey key,
                           const HashEvidence evidence) {
  if (!key.valid()) {
    return true;
  }
  const ReferenceStatus status =
      references.check(backend == Backend::Cpu, key, evidence);
  if (status == ReferenceStatus::Established ||
      status == ReferenceStatus::Matched) {
    return true;
  }
  std::fprintf(stderr,
               "reference %s/%.*s/%.*s/%zu failed: status=%u graph=%llu "
               "output=%llu\n",
               Name(backend), static_cast<int>(key.workload.size()),
               key.workload.data(), static_cast<int>(key.variant.size()),
               key.variant.data(), key.count, static_cast<unsigned>(status),
               static_cast<unsigned long long>(evidence.graph),
               static_cast<unsigned long long>(evidence.output));
  return false;
}

template <class Stats, class Memory>
bool ReportHostMap(const char *const host, const std::size_t count,
                   const double median_us, const Stats &stats,
                   const Memory &memory, const WarmCounters &warm) {
  std::printf("%s,cpu,map,%zu,%.3f", host, count, median_us);
  PrintStats(stats);
  PrintWarm(warm);
  std::printf(",%llu,%llu\n",
              static_cast<unsigned long long>(memory.resident.current),
              static_cast<unsigned long long>(memory.staging.current));
  return warm.zero();
}

template <class ProgramResult, class... Input>
bool Bench(const std::string_view family, const Backend backend,
           ProgramResult &program, const std::size_t iterations,
           const ReferenceKey reference, const Input &...input) {
  if (!program) {
    std::printf("%s,%.*s,unavailable,%.*s\n", Name(backend),
                static_cast<int>(family.size()), family.data(),
                static_cast<int>(program.error().size()),
                program.error().data());
    return false;
  }
  auto job = program->resident(input...);
  if (!job) {
    std::printf("%s,%.*s,resident_failed,%.*s\n", Name(backend),
                static_cast<int>(family.size()), family.data(),
                static_cast<int>(job.error().size()), job.error().data());
    return false;
  }
  if (!job->run()) {
    std::printf("%s,%.*s,warmup_failed\n", Name(backend),
                static_cast<int>(family.size()), family.data());
    return false;
  }
  std::vector<double> samples;
  samples.reserve(iterations);
  WarmCounters warm{};
  for (std::size_t i = 0; i < iterations; ++i) {
    const auto begin = Clock::now();
    const auto status = job->run();
    const auto end = Clock::now();
    if (!status) {
      std::printf("%s,%.*s,run_failed,%.*s\n", Name(backend),
                  static_cast<int>(family.size()), family.data(),
                  static_cast<int>(status.error().size()),
                  status.error().data());
      return false;
    }
    warm.observe(job->stats());
    samples.push_back(
        std::chrono::duration<double, std::micro>(end - begin).count());
  }
  HashEvidence evidence{};
  if (!CaptureOutput(*job, backend, family, evidence)) {
    return false;
  }
  const bool reference_ok = CheckReference(backend, reference, evidence);
  const auto stats = job->stats();
  const auto memory = job->memory();
  std::printf("%s,%.*s,%s,%.3f", Name(backend), static_cast<int>(family.size()),
              family.data(), reference_ok ? "ok" : "reference_failed",
              Median(samples));
  PrintStats(stats);
  PrintWarm(warm);
  std::printf(",%llu,%llu\n",
              static_cast<unsigned long long>(memory.resident.current),
              static_cast<unsigned long long>(memory.staging.current));
  return warm.zero() && reference_ok;
}

} // namespace rund::measure::compute
