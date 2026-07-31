#include "model.hpp"

namespace rund::measure::compute {

void PrintPipelineColumns() {
  std::fputs(
      "pipeline_columns,backend,command_path,status,count,samples,"
      "serial_first,pipeline_first,serial_wall_median_us,"
      "pipeline_wall_median_us,speedup,serial_submit_wait_us,"
      "pipeline_submit_wait_us,serial_kernel_us,pipeline_kernel_us,"
      "pipeline_claim_us,pipeline_control_us,serial_read_median_us,"
      "pipeline_read_median_us,serial_readback_us,pipeline_readback_us,"
      "serial_command_submits,pipeline_command_submits,serial_dispatches,"
      "pipeline_dispatches,step_count,resource_count,barrier_count,"
      "claim_conflict_count,verified_step_count,failed_step_index,"
      "status_entry_count,control_byte_count,control_command_count,"
      "pipeline_fingerprint_hi,pipeline_fingerprint_lo,serial_content_hash,"
      "pipeline_content_hash,"
      "content_parity,warm_pipeline_compiles,warm_buffer_allocations,"
      "warm_descriptor_pool_creations,warm_descriptor_set_allocations,"
      "warm_uploaded_bytes,warm_download_events,warm_downloaded_bytes,"
      "warm_internal_roundtrip_bytes,warm_external_roundtrip_bytes,warm_zero\n",
      stdout);
}

bool MeasurePipeline(const Backend backend, const std::size_t count,
                     const std::size_t samples) {
  if ((backend != Backend::Metal && backend != Backend::Vulkan) ||
      count == 0u || samples == 0u || samples % 2u != 0u) {
    std::fprintf(stderr, "pipeline measurement configuration invalid\n");
    return false;
  }

  std::vector<std::int32_t> input_values(count);
  for (std::size_t index = 0u; index < count; ++index) {
    input_values[index] = static_cast<std::int32_t>(index % 127u) - 63;
  }

  auto device = ::rund::compute::open(TargetFor(backend));
  if (!device) {
    std::fprintf(stderr, "pipeline %s open failed: %.*s\n", Name(backend),
                 static_cast<int>(device.error().size()),
                 device.error().data());
    return false;
  }
  auto first = ::rund::compute::on(*device)
                   .map<std::int32_t>("measure-pipeline-scale", count,
                                      [](auto value) { return value * 3 + 1; })
                   .compile();
  auto second = ::rund::compute::on(*device)
                    .map<std::int32_t>("measure-pipeline-bias", count,
                                       [](auto value) { return value * 5 - 7; })
                    .compile();
  auto third = ::rund::compute::on(*device)
                   .map<std::int32_t>("measure-pipeline-finish", count,
                                      [](auto value) { return value * 2 + 11; })
                   .compile();
  auto input = device->upload<std::int32_t>(input_values);
  auto first_output = device->buffer<std::int32_t>(count);
  auto second_output = device->buffer<std::int32_t>(count);
  auto output = device->buffer<std::int32_t>(count);
  if (!first || !second || !third || !input || !first_output ||
      !second_output || !output) {
    std::fprintf(stderr, "pipeline %s preparation input failed\n",
                 Name(backend));
    return false;
  }
  auto prepared = ::rund::compute::pipeline(*device)
                      .then(*first, ::rund::compute::read(*input),
                            ::rund::compute::write(*first_output))
                      .then(*second, ::rund::compute::read(*first_output),
                            ::rund::compute::write(*second_output))
                      .then(*third, ::rund::compute::read(*second_output),
                            ::rund::compute::write(*output))
                      .prepare();
  if (!prepared || !prepared->fingerprint()) {
    std::fprintf(stderr, "pipeline %s prepare failed\n", Name(backend));
    return false;
  }
  const ::rund::compute::graph::Fingerprint fingerprint =
      prepared->fingerprint();

  Durations serial{};
  Durations pipeline{};
  serial.reserve(samples);
  pipeline.reserve(samples);
  WarmCounters warm{};
  ExecutionCounters serial_counters{};
  ExecutionCounters pipeline_counters{};
  Stats pipeline_stats{};

  const auto run_serial = [&](const bool timed,
                              std::vector<std::int32_t> *const read_values) {
    const auto begin = Clock::now();
    auto first_run = first->run(*input, *first_output);
    auto second_run = second->run(*first_output, *second_output);
    auto third_run = third->run(*second_output, *output);
    const auto end = Clock::now();
    if (!first_run || !second_run || !third_run) {
      std::fprintf(stderr, "pipeline %s serial execution failed\n",
                   Name(backend));
      return false;
    }
    std::uint64_t submits{};
    std::uint64_t dispatches{};
    std::uint64_t submit_ns{};
    std::uint64_t kernel_ns{};
    for (const Stats &stats :
         {first_run->stats(), second_run->stats(), third_run->stats()}) {
      ObserveWarm(warm, stats);
      ::rund::detail::counter::Accumulate(submits, stats.command_submits);
      ::rund::detail::counter::Accumulate(dispatches, stats.dispatches);
      ::rund::detail::counter::Accumulate(submit_ns, stats.submit_wait_ns);
      ::rund::detail::counter::Accumulate(kernel_ns, stats.kernel_ns);
    }
    if (submits != 3u || dispatches != 3u) {
      std::fprintf(stderr, "pipeline %s serial counters failed\n",
                   Name(backend));
      return false;
    }
    serial_counters = {.command_submits = submits, .dispatches = dispatches};
    if (timed) {
      serial.wall.push_back(
          std::chrono::duration<double, std::micro>(end - begin).count());
      serial.submit.push_back(static_cast<double>(submit_ns) / 1'000.0);
      serial.kernel.push_back(static_cast<double>(kernel_ns) / 1'000.0);
    }
    if (read_values != nullptr) {
      const auto read_begin = Clock::now();
      const auto read_status =
          third_run->read(*output, std::span<std::int32_t>{*read_values});
      const auto read_end = Clock::now();
      if (!read_status || !ValidValues(input_values, *read_values)) {
        std::fprintf(stderr, "pipeline %s serial read failed\n", Name(backend));
        return false;
      }
      serial.read.push_back(
          std::chrono::duration<double, std::micro>(read_end - read_begin)
              .count());
      serial.readback.push_back(
          static_cast<double>(third_run->stats().readback_ns) / 1'000.0);
    }
    return true;
  };

  const auto run_pipeline = [&](const bool timed,
                                std::vector<std::int32_t> *const read_values) {
    const auto begin = Clock::now();
    const auto status = prepared->run();
    const auto end = Clock::now();
    if (!status) {
      std::fprintf(stderr, "pipeline %s execution failed: %.*s\n",
                   Name(backend), static_cast<int>(status.error().size()),
                   status.error().data());
      return false;
    }
    pipeline_stats = prepared->stats();
    ObserveWarm(warm, pipeline_stats);
    if (!PipelineCounters(backend, pipeline_stats, fingerprint.lo)) {
      std::fprintf(stderr, "pipeline %s counters failed\n", Name(backend));
      return false;
    }
    pipeline_counters = {
        .command_submits = pipeline_stats.command_submits,
        .dispatches = pipeline_stats.dispatches,
    };
    if (timed) {
      pipeline.wall.push_back(
          std::chrono::duration<double, std::micro>(end - begin).count());
      pipeline.submit.push_back(
          static_cast<double>(pipeline_stats.submit_wait_ns) / 1'000.0);
      pipeline.kernel.push_back(static_cast<double>(pipeline_stats.kernel_ns) /
                                1'000.0);
      pipeline.claim.push_back(
          static_cast<double>(pipeline_stats.pipeline.claim_ns) / 1'000.0);
      pipeline.control.push_back(
          static_cast<double>(pipeline_stats.pipeline.control_ns) / 1'000.0);
    }
    if (read_values != nullptr) {
      const auto read_begin = Clock::now();
      const auto read_status =
          prepared->read(*output, std::span<std::int32_t>{*read_values});
      const auto read_end = Clock::now();
      if (!read_status || !ValidValues(input_values, *read_values)) {
        std::fprintf(stderr, "pipeline %s read failed\n", Name(backend));
        return false;
      }
      pipeline.read.push_back(
          std::chrono::duration<double, std::micro>(read_end - read_begin)
              .count());
      pipeline.readback.push_back(
          static_cast<double>(prepared->stats().readback_ns) / 1'000.0);
    }
    return true;
  };

  if (!run_serial(false, nullptr) || !run_pipeline(false, nullptr)) {
    return false;
  }

  std::size_t serial_first{};
  std::size_t pipeline_first{};
  for (std::size_t sample = 0u; sample < samples; ++sample) {
    const bool serial_then_pipeline = sample % 2u == 0u;
    const bool ok =
        serial_then_pipeline
            ? run_serial(true, nullptr) && run_pipeline(true, nullptr)
            : run_pipeline(true, nullptr) && run_serial(true, nullptr);
    if (!ok) {
      return false;
    }
    serial_first += serial_then_pipeline ? 1u : 0u;
    pipeline_first += serial_then_pipeline ? 0u : 1u;
  }

  std::vector<std::int32_t> serial_values(count);
  std::vector<std::int32_t> pipeline_values(count);
  std::uint64_t serial_hash{};
  std::uint64_t pipeline_hash{};
  for (std::size_t sample = 0u; sample < samples; ++sample) {
    const bool serial_then_pipeline = sample % 2u == 0u;
    const bool ok = serial_then_pipeline
                        ? run_serial(false, &serial_values) &&
                              run_pipeline(false, &pipeline_values)
                        : run_pipeline(false, &pipeline_values) &&
                              run_serial(false, &serial_values);
    if (!ok) {
      return false;
    }
    const std::uint64_t current_serial = ContentHash(serial_values);
    const std::uint64_t current_pipeline = ContentHash(pipeline_values);
    if (current_serial == 0u || current_serial != current_pipeline ||
        (serial_hash != 0u && serial_hash != current_serial) ||
        (pipeline_hash != 0u && pipeline_hash != current_pipeline)) {
      std::fprintf(stderr, "pipeline %s content parity failed\n",
                   Name(backend));
      return false;
    }
    serial_hash = current_serial;
    pipeline_hash = current_pipeline;
  }

  const bool balanced =
      serial_first == samples / 2u && pipeline_first == samples / 2u;
  const bool parity = serial_hash != 0u && serial_hash == pipeline_hash;
  const bool contract = balanced && parity && warm.zero();
  const double serial_wall_us = Median(serial.wall);
  const double pipeline_wall_us = Median(pipeline.wall);
  const double speedup =
      pipeline_wall_us == 0.0 ? 0.0 : serial_wall_us / pipeline_wall_us;
  const auto &evidence = pipeline_stats.pipeline;

  std::printf(
      "pipeline,%s,%s,%s,%zu,%zu,%zu,%zu,%.3f,%.3f,%.6f,%.3f,%.3f,"
      "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%llu,%llu,%llu,%llu,"
      "%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,"
      "%llu,%u,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%u\n",
      Name(backend), CommandPath(backend), contract ? "ok" : "contract_failed",
      count, samples, serial_first, pipeline_first, serial_wall_us,
      pipeline_wall_us, speedup, Median(serial.submit), Median(pipeline.submit),
      Median(serial.kernel), Median(pipeline.kernel), Median(pipeline.claim),
      Median(pipeline.control), Median(serial.read), Median(pipeline.read),
      Median(serial.readback), Median(pipeline.readback),
      static_cast<unsigned long long>(serial_counters.command_submits),
      static_cast<unsigned long long>(pipeline_counters.command_submits),
      static_cast<unsigned long long>(serial_counters.dispatches),
      static_cast<unsigned long long>(pipeline_counters.dispatches),
      static_cast<unsigned long long>(evidence.step_count),
      static_cast<unsigned long long>(evidence.resource_count),
      static_cast<unsigned long long>(evidence.barrier_count),
      static_cast<unsigned long long>(evidence.claim_conflict_count),
      static_cast<unsigned long long>(evidence.verified_step_count),
      static_cast<unsigned long long>(evidence.failed_step_index),
      static_cast<unsigned long long>(evidence.status_entry_count),
      static_cast<unsigned long long>(evidence.control_byte_count),
      static_cast<unsigned long long>(evidence.control_command_count),
      static_cast<unsigned long long>(fingerprint.hi),
      static_cast<unsigned long long>(fingerprint.lo),
      static_cast<unsigned long long>(serial_hash),
      static_cast<unsigned long long>(pipeline_hash), parity ? 1u : 0u,
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
