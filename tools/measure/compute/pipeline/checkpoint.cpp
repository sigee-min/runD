#include "model.hpp"

#include "allocation.hpp"

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace rund::measure::compute {
namespace {

[[nodiscard]] std::uint64_t MaximumResidentBytes() noexcept {
  rusage usage{};
  if (::getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss < 0) {
    return 0u;
  }
  const auto rss = static_cast<std::uint64_t>(usage.ru_maxrss);
#if defined(__APPLE__)
  return rss;
#else
  return rss > std::numeric_limits<std::uint64_t>::max() / 1024u
             ? std::numeric_limits<std::uint64_t>::max()
             : rss * 1024u;
#endif
}

[[nodiscard]] std::uint64_t Delta(const std::uint64_t after,
                                  const std::uint64_t before) noexcept {
  return after >= before ? after - before : 0u;
}

struct CheckpointObservation final {
  std::vector<double> ticks;
  std::vector<double> exports;
  std::vector<double> run_allocation_counts;
  std::vector<double> run_allocation_bytes;
  std::vector<double> path_allocation_counts;
  std::vector<double> path_allocation_bytes;
  ::rund::compute::Stats stats{};
  ::rund::compute::CheckpointStats checkpoint{};
  ::rund::compute::MemoryStats memory_before{};
  ::rund::compute::MemoryStats memory_after{};
  ::rund::compute::graph::Fingerprint fingerprint{};
  std::uint64_t generation{};
  std::uint64_t hash{};
  std::uint64_t rss_before{};
  std::uint64_t rss_after{};
};

void PrintObservation(const Backend backend, const char *const path,
                      const std::size_t count, const std::size_t samples,
                      const bool ok, CheckpointObservation &observed) {
  const std::uint64_t payload_bytes =
      static_cast<std::uint64_t>(count) * sizeof(std::int32_t);
  const double export_wall =
      observed.exports.empty() ? 0.0 : Median(observed.exports);
  const double run_allocation_count =
      observed.run_allocation_counts.empty()
          ? 0.0
          : Median(observed.run_allocation_counts);
  const double run_allocation_bytes =
      observed.run_allocation_bytes.empty()
          ? 0.0
          : Median(observed.run_allocation_bytes);
  const double path_allocation_count =
      observed.path_allocation_counts.empty()
          ? 0.0
          : Median(observed.path_allocation_counts);
  const double path_allocation_bytes =
      observed.path_allocation_bytes.empty()
          ? 0.0
          : Median(observed.path_allocation_bytes);
  std::printf(
      "checkpoint,%s,%s,%s,%zu,%zu,%llu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
      "%llu,%llu,%llu,%llu,%llu,"
      "%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,"
      "%llu,%llu,%llu,%llu,%llu,%llu\n",
      Name(backend), path, ok ? "ok" : "contract_failed", count, samples,
      static_cast<unsigned long long>(payload_bytes), Median(observed.ticks),
      export_wall, run_allocation_count, run_allocation_bytes,
      path_allocation_count, path_allocation_bytes,
      static_cast<unsigned long long>(observed.generation),
      static_cast<unsigned long long>(observed.hash),
      static_cast<unsigned long long>(observed.fingerprint.hi),
      static_cast<unsigned long long>(observed.fingerprint.lo),
      static_cast<unsigned long long>(
          observed.checkpoint.device_state_acquire_count),
      static_cast<unsigned long long>(
          observed.checkpoint.device_state_rebase_count),
      static_cast<unsigned long long>(
          observed.checkpoint.device_state_copy_byte_count),
      static_cast<unsigned long long>(
          observed.checkpoint.device_state_copy_command_count),
      static_cast<unsigned long long>(
          observed.checkpoint.reusable_snapshot_count),
      static_cast<unsigned long long>(
          observed.checkpoint.reusable_snapshot_byte_count),
      static_cast<unsigned long long>(
          observed.checkpoint.reusable_snapshot_transfer_count),
      static_cast<unsigned long long>(observed.stats.uploaded_bytes),
      static_cast<unsigned long long>(observed.stats.download_events),
      static_cast<unsigned long long>(observed.stats.downloaded_bytes),
      static_cast<unsigned long long>(
          Delta(observed.memory_after.staging.cumulative,
                observed.memory_before.staging.cumulative)),
      static_cast<unsigned long long>(observed.memory_after.staging.peak),
      static_cast<unsigned long long>(
          Delta(observed.memory_after.staging.reused,
                observed.memory_before.staging.reused)),
      static_cast<unsigned long long>(observed.stats.buffer_allocations),
      static_cast<unsigned long long>(observed.stats.buffer_reuses),
      static_cast<unsigned long long>(observed.stats.command_submits),
      static_cast<unsigned long long>(observed.rss_before),
      static_cast<unsigned long long>(observed.rss_after),
      static_cast<unsigned long long>(
          Delta(observed.rss_after, observed.rss_before)));
}

} // namespace

void PrintCheckpointColumns() {
  std::fputs(
      "checkpoint_columns,backend,path,status,count,samples,payload_bytes,"
      "tick_path_wall_median_us,export_copy_hash_wall_median_us,"
      "run_process_allocation_count_median,run_process_allocation_bytes_median,"
      "checkpoint_process_allocation_count_median,"
      "checkpoint_process_allocation_bytes_median,generation,hash,"
      "fingerprint_hi,fingerprint_lo,"
      "device_acquires,device_rebases,device_copy_bytes,device_copy_commands,"
      "reusable_exports,reusable_bytes,reusable_transfers,uploaded_bytes,"
      "download_events,downloaded_bytes,staging_cumulative_delta,"
      "staging_peak,staging_reused_delta,backend_buffer_allocations,"
      "backend_buffer_reuses,command_submits,rss_before_bytes,rss_after_bytes,"
      "rss_high_water_delta_bytes\n",
      stdout);
}

bool MeasureCheckpoints(const Backend backend, const std::size_t count,
                        const std::size_t samples) {
  using namespace ::rund::compute;
  if (backend == Backend::Unavailable || count == 0u || samples == 0u ||
      count >
          std::numeric_limits<std::uint64_t>::max() / sizeof(std::int32_t)) {
    std::fputs("checkpoint measurement configuration invalid\n", stderr);
    return false;
  }
  std::vector<std::int32_t> initial(count, 0);
  auto device = open(TargetFor(backend));
  if (!device) {
    std::fprintf(stderr, "checkpoint %s open failed: %.*s\n", Name(backend),
                 static_cast<int>(device.error().size()),
                 device.error().data());
    return false;
  }
  auto advance = on(*device)
                     .map<std::int32_t>("measure-pipeline-checkpoint", count,
                                        [](auto value) { return value + 1; })
                     .compile();
  if (!advance) {
    std::fprintf(stderr, "checkpoint %s compile failed\n", Name(backend));
    return false;
  }
  const auto prepare = [&]() {
    auto first =
        device->upload<std::int32_t>(std::span<const std::int32_t>{initial});
    auto second = device->buffer<std::int32_t>(count);
    if (!first || !second) {
      return Result<Pipeline>::fail(Reason::PipelineInvalid);
    }
    return pipeline(*device)
        .state(*first, *second)
        .then(*advance, read(*first), write(*second))
        .commit()
        .prepare();
  };

  auto latest_pipeline = prepare();
  auto reusable_pipeline = prepare();
  auto immutable_pipeline = prepare();
  auto storage_result =
      reusable_pipeline
          ? reusable_pipeline->snapshot_storage()
          : Result<SnapshotStorage>::fail(Reason::PipelineInvalid);
  if (!latest_pipeline || !reusable_pipeline || !immutable_pipeline ||
      !storage_result) {
    std::fprintf(stderr, "checkpoint %s prepare failed\n", Name(backend));
    return false;
  }
  SnapshotStorage storage = std::move(storage_result).value();
  if (!latest_pipeline->run() || !reusable_pipeline->run() ||
      !reusable_pipeline->snapshot_into(storage) ||
      !reusable_pipeline->snapshot_into(storage) ||
      !immutable_pipeline->run() || !immutable_pipeline->snapshot()) {
    std::fprintf(stderr, "checkpoint %s warm-up failed\n", Name(backend));
    return false;
  }

  CheckpointObservation latest_observation{};
  latest_observation.ticks.reserve(samples);
  latest_observation.run_allocation_counts.reserve(samples);
  latest_observation.run_allocation_bytes.reserve(samples);
  latest_observation.path_allocation_counts.reserve(1u);
  latest_observation.path_allocation_bytes.reserve(1u);
  node_compute_allocation::Start();
  auto latest = latest_pipeline->latest_device_state();
  node_compute_allocation::Stop();
  latest_observation.path_allocation_counts.push_back(
      static_cast<double>(node_compute_allocation::Count()));
  latest_observation.path_allocation_bytes.push_back(
      static_cast<double>(node_compute_allocation::Bytes()));
  if (!latest) {
    return false;
  }
  latest_observation.memory_before = latest_pipeline->memory();
  latest_observation.rss_before = MaximumResidentBytes();
  for (std::size_t sample = 0u; sample < samples; ++sample) {
    node_compute_allocation::Start();
    const auto begin = Clock::now();
    const Status ran = latest_pipeline->run();
    const auto end = Clock::now();
    node_compute_allocation::Stop();
    if (!ran) {
      return false;
    }
    latest_observation.ticks.push_back(
        std::chrono::duration<double, std::micro>(end - begin).count());
    latest_observation.run_allocation_counts.push_back(
        static_cast<double>(node_compute_allocation::Count()));
    latest_observation.run_allocation_bytes.push_back(
        static_cast<double>(node_compute_allocation::Bytes()));
  }
  latest_observation.rss_after = MaximumResidentBytes();
  latest_observation.memory_after = latest_pipeline->memory();
  latest_observation.stats = latest_pipeline->stats();
  latest_observation.checkpoint = latest_pipeline->checkpoint_stats();
  latest_observation.fingerprint = latest_pipeline->fingerprint();
  latest_observation.generation = latest->generation();
  const bool latest_ok =
      latest->valid() && latest_observation.generation == samples + 1u &&
      latest_observation.stats.uploaded_bytes == 0u &&
      latest_observation.stats.downloaded_bytes == 0u &&
      latest_observation.stats.download_events == 0u &&
      latest_observation.checkpoint.device_state_acquire_count == 1u;
  PrintObservation(backend, "latest_device", count, samples, latest_ok,
                   latest_observation);

  CheckpointObservation reusable_observation{};
  reusable_observation.ticks.reserve(samples);
  reusable_observation.exports.reserve(samples);
  reusable_observation.run_allocation_counts.reserve(samples);
  reusable_observation.run_allocation_bytes.reserve(samples);
  reusable_observation.path_allocation_counts.reserve(samples);
  reusable_observation.path_allocation_bytes.reserve(samples);
  const CheckpointStats reusable_before = reusable_pipeline->checkpoint_stats();
  reusable_observation.memory_before = reusable_pipeline->memory();
  reusable_observation.rss_before = MaximumResidentBytes();
  for (std::size_t sample = 0u; sample < samples; ++sample) {
    node_compute_allocation::Start();
    const auto run_begin = Clock::now();
    const Status ran = reusable_pipeline->run();
    const auto run_end = Clock::now();
    node_compute_allocation::Stop();
    const double run_wall =
        std::chrono::duration<double, std::micro>(run_end - run_begin).count();
    reusable_observation.run_allocation_counts.push_back(
        static_cast<double>(node_compute_allocation::Count()));
    reusable_observation.run_allocation_bytes.push_back(
        static_cast<double>(node_compute_allocation::Bytes()));
    node_compute_allocation::Start();
    const auto export_begin = Clock::now();
    const Status saved = ran ? reusable_pipeline->snapshot_into(storage) : ran;
    const auto export_end = Clock::now();
    node_compute_allocation::Stop();
    if (!saved) {
      return false;
    }
    const double export_wall =
        std::chrono::duration<double, std::micro>(export_end - export_begin)
            .count();
    reusable_observation.ticks.push_back(run_wall + export_wall);
    reusable_observation.exports.push_back(export_wall);
    reusable_observation.path_allocation_counts.push_back(
        static_cast<double>(node_compute_allocation::Count()));
    reusable_observation.path_allocation_bytes.push_back(
        static_cast<double>(node_compute_allocation::Bytes()));
  }
  reusable_observation.rss_after = MaximumResidentBytes();
  reusable_observation.memory_after = reusable_pipeline->memory();
  reusable_observation.stats = reusable_pipeline->stats();
  const CheckpointStats reusable_after = reusable_pipeline->checkpoint_stats();
  reusable_observation.checkpoint = CheckpointStats{
      .reusable_snapshot_count = Delta(reusable_after.reusable_snapshot_count,
                                       reusable_before.reusable_snapshot_count),
      .reusable_snapshot_byte_count =
          Delta(reusable_after.reusable_snapshot_byte_count,
                reusable_before.reusable_snapshot_byte_count),
      .reusable_snapshot_hash = reusable_after.reusable_snapshot_hash,
      .reusable_snapshot_transfer_count =
          Delta(reusable_after.reusable_snapshot_transfer_count,
                reusable_before.reusable_snapshot_transfer_count),
  };
  reusable_observation.fingerprint = storage.fingerprint();
  reusable_observation.generation = storage.generation();
  reusable_observation.hash = storage.hash();
  const std::uint64_t payload_bytes =
      static_cast<std::uint64_t>(count) * sizeof(std::int32_t);
  const bool reusable_ok =
      storage.has_snapshot() &&
      reusable_observation.generation == samples + 1u &&
      reusable_observation.hash != 0u &&
      reusable_observation.checkpoint.reusable_snapshot_count == samples &&
      reusable_observation.checkpoint.reusable_snapshot_byte_count ==
          payload_bytes * samples &&
      reusable_observation.stats.uploaded_bytes == 0u &&
      reusable_observation.stats.downloaded_bytes ==
          (backend == Backend::Cpu ? 0u : payload_bytes) &&
      reusable_observation.stats.download_events ==
          (backend == Backend::Cpu ? 0u : 1u);
  PrintObservation(backend, "reusable_host", count, samples, reusable_ok,
                   reusable_observation);

  CheckpointObservation immutable_observation{};
  immutable_observation.ticks.reserve(samples);
  immutable_observation.exports.reserve(samples);
  immutable_observation.run_allocation_counts.reserve(samples);
  immutable_observation.run_allocation_bytes.reserve(samples);
  immutable_observation.path_allocation_counts.reserve(samples);
  immutable_observation.path_allocation_bytes.reserve(samples);
  std::vector<StateSnapshot> retained;
  retained.reserve(samples);
  const std::uint64_t immutable_snapshot_bytes_before =
      immutable_pipeline->stats().publication.snapshot_byte_count;
  immutable_observation.memory_before = immutable_pipeline->memory();
  immutable_observation.rss_before = MaximumResidentBytes();
  for (std::size_t sample = 0u; sample < samples; ++sample) {
    node_compute_allocation::Start();
    const auto run_begin = Clock::now();
    const Status ran = immutable_pipeline->run();
    const auto run_end = Clock::now();
    node_compute_allocation::Stop();
    const double run_wall =
        std::chrono::duration<double, std::micro>(run_end - run_begin).count();
    immutable_observation.run_allocation_counts.push_back(
        static_cast<double>(node_compute_allocation::Count()));
    immutable_observation.run_allocation_bytes.push_back(
        static_cast<double>(node_compute_allocation::Bytes()));
    node_compute_allocation::Start();
    const auto export_begin = Clock::now();
    auto saved = ran ? immutable_pipeline->snapshot()
                     : Result<StateSnapshot>::fail(ran.reason());
    const auto export_end = Clock::now();
    node_compute_allocation::Stop();
    if (!saved) {
      return false;
    }
    retained.push_back(std::move(saved).value());
    const double export_wall =
        std::chrono::duration<double, std::micro>(export_end - export_begin)
            .count();
    immutable_observation.ticks.push_back(run_wall + export_wall);
    immutable_observation.exports.push_back(export_wall);
    immutable_observation.path_allocation_counts.push_back(
        static_cast<double>(node_compute_allocation::Count()));
    immutable_observation.path_allocation_bytes.push_back(
        static_cast<double>(node_compute_allocation::Bytes()));
  }
  immutable_observation.rss_after = MaximumResidentBytes();
  immutable_observation.memory_after = immutable_pipeline->memory();
  immutable_observation.stats = immutable_pipeline->stats();
  immutable_observation.checkpoint = immutable_pipeline->checkpoint_stats();
  immutable_observation.fingerprint = retained.back().fingerprint();
  immutable_observation.generation = retained.back().generation();
  immutable_observation.hash = retained.back().hash();
  const bool immutable_ok =
      retained.size() == samples &&
      immutable_observation.generation == samples + 1u &&
      immutable_observation.hash != 0u &&
      Delta(immutable_observation.stats.publication.snapshot_byte_count,
            immutable_snapshot_bytes_before) == payload_bytes * samples &&
      immutable_observation.stats.uploaded_bytes == 0u &&
      immutable_observation.stats.downloaded_bytes ==
          (backend == Backend::Cpu ? 0u : payload_bytes) &&
      immutable_observation.stats.download_events ==
          (backend == Backend::Cpu ? 0u : 1u);
  PrintObservation(backend, "immutable_host", count, samples, immutable_ok,
                   immutable_observation);
  return latest_ok && reusable_ok && immutable_ok;
}

} // namespace rund::measure::compute
