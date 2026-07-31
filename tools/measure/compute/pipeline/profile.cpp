#include "model.hpp"

namespace rund::measure::compute {

void PrintPipelineProfileColumns() {
  std::fputs(
      "pipeline_profile_columns,backend,command_path,status,count,samples,"
      "abba_cycles,disabled_wall_median_us,enabled_wall_median_us,"
      "delta_direction,wall_delta_us,enabled_over_disabled,"
      "observation_median_us,disabled_command_submits,"
      "enabled_command_submits,disabled_dispatches,enabled_dispatches,"
      "step_count,profile_rows,step_clock,instrumentation_command_count,"
      "profile_workgroups,profile_work_items,"
      "instrumentation_byte_count,fingerprint_hi,fingerprint_lo,"
      "disabled_content_hash,enabled_content_hash,output_parity,"
      "fingerprint_parity,backend_parity,failure_identity_parity,"
      "topology_parity,profile_parity,semantic_parity,"
      "warm_pipeline_compiles,warm_buffer_allocations,"
      "warm_descriptor_pool_creations,warm_descriptor_set_allocations,"
      "warm_uploaded_bytes,warm_download_events,warm_downloaded_bytes,"
      "warm_internal_roundtrip_bytes,warm_external_roundtrip_bytes,"
      "warm_zero\n",
      stdout);
}

bool MeasurePipelineProfile(const Backend backend, const std::size_t count,
                            const std::size_t samples) {
  using ::rund::compute::Pipeline;
  using ::rund::compute::PipelineProfile;
  using ::rund::compute::PipelineProfileSnapshot;
  using ::rund::compute::PipelineStepProfile;
  using ::rund::compute::Reason;
  using ::rund::compute::Status;
  if ((backend != Backend::Metal && backend != Backend::Vulkan) ||
      count == 0u || samples == 0u || samples % 2u != 0u) {
    std::fprintf(stderr,
                 "pipeline profile measurement configuration invalid\n");
    return false;
  }

  std::vector<std::int32_t> input_values(count);
  for (std::size_t index = 0u; index < count; ++index) {
    input_values[index] = static_cast<std::int32_t>(index % 127u) - 63;
  }
  auto device = ::rund::compute::open(TargetFor(backend));
  if (!device) {
    std::fprintf(stderr, "pipeline profile %s open failed: %.*s\n",
                 Name(backend), static_cast<int>(device.error().size()),
                 device.error().data());
    return false;
  }
  auto first = ::rund::compute::on(*device)
                   .map<std::int32_t>("measure-pipeline-profile-scale", count,
                                      [](auto value) { return value * 3 + 1; })
                   .compile();
  auto second = ::rund::compute::on(*device)
                    .map<std::int32_t>("measure-pipeline-profile-bias", count,
                                       [](auto value) { return value * 5 - 7; })
                    .compile();
  auto third = ::rund::compute::on(*device)
                   .map<std::int32_t>("measure-pipeline-profile-finish", count,
                                      [](auto value) { return value * 2 + 11; })
                   .compile();
  auto input = device->upload<std::int32_t>(input_values);
  auto disabled_first = device->buffer<std::int32_t>(count);
  auto disabled_second = device->buffer<std::int32_t>(count);
  auto disabled_output = device->buffer<std::int32_t>(count);
  auto enabled_first = device->buffer<std::int32_t>(count);
  auto enabled_second = device->buffer<std::int32_t>(count);
  auto enabled_output = device->buffer<std::int32_t>(count);
  if (!first || !second || !third || !input || !disabled_first ||
      !disabled_second || !disabled_output || !enabled_first ||
      !enabled_second || !enabled_output) {
    std::fprintf(stderr, "pipeline profile %s preparation input failed\n",
                 Name(backend));
    return false;
  }
  auto disabled = ::rund::compute::pipeline(*device)
                      .profile(PipelineProfile::None)
                      .then(*first, ::rund::compute::read(*input),
                            ::rund::compute::write(*disabled_first))
                      .then(*second, ::rund::compute::read(*disabled_first),
                            ::rund::compute::write(*disabled_second))
                      .then(*third, ::rund::compute::read(*disabled_second),
                            ::rund::compute::write(*disabled_output))
                      .prepare();
  auto enabled = ::rund::compute::pipeline(*device)
                     .profile(PipelineProfile::Steps)
                     .then(*first, ::rund::compute::read(*input),
                           ::rund::compute::write(*enabled_first))
                     .then(*second, ::rund::compute::read(*enabled_first),
                           ::rund::compute::write(*enabled_second))
                     .then(*third, ::rund::compute::read(*enabled_second),
                           ::rund::compute::write(*enabled_output))
                     .prepare();
  if (!disabled || !enabled || !disabled->fingerprint() ||
      disabled->fingerprint() != enabled->fingerprint()) {
    std::fprintf(stderr, "pipeline profile %s prepare parity failed\n",
                 Name(backend));
    return false;
  }
  const auto fingerprint = disabled->fingerprint();
  const std::array program_fingerprints{
      first->fingerprint(), second->fingerprint(), third->fingerprint()};
  std::array<PipelineStepProfile, 3u> rows{};
  const auto unavailable = disabled->profile(rows);
  if (unavailable || unavailable.reason() != Reason::ProfileUnavailable) {
    std::fprintf(stderr, "pipeline profile %s disabled surface failed\n",
                 Name(backend));
    return false;
  }

  std::vector<double> disabled_wall{};
  std::vector<double> enabled_wall{};
  std::vector<double> observation{};
  disabled_wall.reserve(samples);
  enabled_wall.reserve(samples);
  observation.reserve(samples);
  WarmCounters disabled_warm{};
  WarmCounters enabled_warm{};
  Stats disabled_stats{};
  Stats enabled_stats{};
  Status disabled_status = Status::success();
  Status enabled_status = Status::success();
  PipelineProfileSnapshot enabled_profile{};

  const auto run = [&](Pipeline &pipeline, const bool profiled,
                       const bool timed) {
    const auto begin = Clock::now();
    const Status status = pipeline.run();
    const auto end = Clock::now();
    if (!status) {
      std::fprintf(stderr, "pipeline profile %s %s run failed: %.*s\n",
                   Name(backend), profiled ? "enabled" : "disabled",
                   static_cast<int>(status.error().size()),
                   status.error().data());
      return false;
    }
    Stats &stats = profiled ? enabled_stats : disabled_stats;
    WarmCounters &warm = profiled ? enabled_warm : disabled_warm;
    Status &identity = profiled ? enabled_status : disabled_status;
    stats = pipeline.stats();
    identity = status;
    ObserveWarm(warm, stats);
    if (!PipelineCounters(backend, stats, fingerprint.lo)) {
      std::fprintf(stderr, "pipeline profile %s %s topology failed\n",
                   Name(backend), profiled ? "enabled" : "disabled");
      return false;
    }
    if (timed) {
      (profiled ? enabled_wall : disabled_wall)
          .push_back(
              std::chrono::duration<double, std::micro>(end - begin).count());
    }
    if (!profiled) {
      return true;
    }
    const auto observed = pipeline.profile(rows);
    if (!observed ||
        !ProfileEvidenceValid(backend, *observed, rows, program_fingerprints,
                              fingerprint.lo, count)) {
      std::fprintf(stderr, "pipeline profile %s evidence failed\n",
                   Name(backend));
      return false;
    }
    enabled_profile = *observed;
    if (timed) {
      observation.push_back(
          static_cast<double>(observed->observation.duration_ns) / 1'000.0);
    }
    return true;
  };

  if (!run(*disabled, false, false) || !run(*enabled, true, false)) {
    return false;
  }
  const std::size_t cycles = samples / 2u;
  for (std::size_t cycle = 0u; cycle < cycles; ++cycle) {
    if (!run(*disabled, false, true) || !run(*enabled, true, true) ||
        !run(*enabled, true, true) || !run(*disabled, false, true)) {
      return false;
    }
  }
  if (disabled_wall.size() != samples || enabled_wall.size() != samples ||
      observation.size() != samples) {
    std::fprintf(stderr, "pipeline profile %s ABBA balance failed\n",
                 Name(backend));
    return false;
  }

  std::vector<std::int32_t> disabled_values(count);
  std::vector<std::int32_t> enabled_values(count);
  const Status disabled_read =
      disabled->read(*disabled_output, disabled_values);
  const Status enabled_read = enabled->read(*enabled_output, enabled_values);
  if (!disabled_read || !enabled_read ||
      !ValidValues(input_values, disabled_values) ||
      !ValidValues(input_values, enabled_values)) {
    std::fprintf(stderr, "pipeline profile %s output validation failed\n",
                 Name(backend));
    return false;
  }

  const std::uint64_t disabled_hash = ContentHash(disabled_values);
  const std::uint64_t enabled_hash = ContentHash(enabled_values);
  const bool output_parity =
      disabled_hash != 0u && disabled_hash == enabled_hash;
  const bool fingerprint_parity =
      disabled->fingerprint() == enabled->fingerprint();
  const bool backend_parity =
      disabled_stats.backend == backend && enabled_stats.backend == backend;
  const bool failure_identity_parity =
      disabled_status.ok() && enabled_status.ok() &&
      SameStatusIdentity(disabled_status, enabled_status);
  const bool topology_parity = disabled_stats.command_submits == 1u &&
                               enabled_stats.command_submits == 1u &&
                               disabled_stats.dispatches == 3u &&
                               enabled_stats.dispatches == 3u;
  const bool profile_parity =
      ProfileEvidenceValid(backend, enabled_profile, rows, program_fingerprints,
                           fingerprint.lo, count);
  WarmCounters warm{};
  AccumulateWarm(warm, disabled_warm);
  AccumulateWarm(warm, enabled_warm);
  const bool semantic_parity = output_parity && fingerprint_parity &&
                               backend_parity && failure_identity_parity &&
                               topology_parity && profile_parity;
  const bool contract = semantic_parity && warm.zero();

  const double disabled_us = Median(disabled_wall);
  const double enabled_us = Median(enabled_wall);
  const bool enabled_slower = enabled_us > disabled_us;
  const bool equal = enabled_us == disabled_us;
  const double delta_us =
      enabled_slower ? enabled_us - disabled_us : disabled_us - enabled_us;
  const double ratio = disabled_us == 0.0 ? 0.0 : enabled_us / disabled_us;
  const double observation_us = Median(observation);
  const char *const direction = equal            ? "equal"
                                : enabled_slower ? "enabled_slower"
                                                 : "enabled_faster";
  std::uint64_t profile_workgroups = 0u;
  std::uint64_t profile_work_items = 0u;
  for (const PipelineStepProfile &row : rows) {
    profile_workgroups += row.execution.workgroup_count;
    profile_work_items += row.execution.work_item_count;
  }

  std::printf(
      "pipeline_profile,%s,%s,%s,%zu,%zu,%zu,%.3f,%.3f,%s,%.3f,%.6f,"
      "%.3f,%llu,%llu,%llu,%llu,%llu,%zu,%s,%llu,%llu,%llu,%llu,%llu,"
      "%llu,%llu,"
      "%llu,%u,%u,%u,%u,%u,%u,%u,%llu,%llu,%llu,%llu,%llu,%llu,%llu,"
      "%llu,%llu,%u\n",
      Name(backend), CommandPath(backend),
      contract ? "ok" : "contract_failed", count, samples, cycles, disabled_us,
      enabled_us, direction, delta_us, ratio, observation_us,
      static_cast<unsigned long long>(disabled_stats.command_submits),
      static_cast<unsigned long long>(enabled_stats.command_submits),
      static_cast<unsigned long long>(disabled_stats.dispatches),
      static_cast<unsigned long long>(enabled_stats.dispatches),
      static_cast<unsigned long long>(
          enabled_profile.execution.pipeline.step_count),
      rows.size(), ProfileClockName(rows),
      static_cast<unsigned long long>(
          enabled_profile.instrumentation_command_count),
      static_cast<unsigned long long>(profile_workgroups),
      static_cast<unsigned long long>(profile_work_items),
      static_cast<unsigned long long>(
          enabled_profile.instrumentation_byte_count),
      static_cast<unsigned long long>(fingerprint.hi),
      static_cast<unsigned long long>(fingerprint.lo),
      static_cast<unsigned long long>(disabled_hash),
      static_cast<unsigned long long>(enabled_hash), output_parity ? 1u : 0u,
      fingerprint_parity ? 1u : 0u, backend_parity ? 1u : 0u,
      failure_identity_parity ? 1u : 0u, topology_parity ? 1u : 0u,
      profile_parity ? 1u : 0u, semantic_parity ? 1u : 0u,
      static_cast<unsigned long long>(warm.pipeline_compiles),
      static_cast<unsigned long long>(warm.buffer_allocations),
      static_cast<unsigned long long>(warm.descriptor_pool_creations),
      static_cast<unsigned long long>(warm.descriptor_set_allocations),
      static_cast<unsigned long long>(warm.uploaded_bytes),
      static_cast<unsigned long long>(warm.download_events),
      static_cast<unsigned long long>(warm.downloaded_bytes),
      static_cast<unsigned long long>(warm.internal_roundtrip_bytes),
      static_cast<unsigned long long>(warm.external_roundtrip_bytes),
      warm.zero() ? 1u : 0u);
  return contract;
}

} // namespace rund::measure::compute
