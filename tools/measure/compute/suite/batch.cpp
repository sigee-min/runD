#include "core.hpp"

namespace rund::measure::compute {

[[nodiscard]] bool
BatchJobSharedZero(const rund::compute::Stats &stats) noexcept {
  return stats.command_submits == 0u && stats.command_capacity == 0u &&
         stats.command_inflight_peak == 0u &&
         stats.command_capacity_rejections == 0u && stats.kernel_ns == 0u &&
         stats.kernel_samples == 0u && stats.submit_wait_ns == 0u;
}

bool BatchJobs(const Backend backend, const std::size_t count,
               const std::size_t samples) {
  constexpr std::size_t JobCount = rund::compute::Batch::capacity();
  static_assert(JobCount == 64u);
  if (samples == 0u || samples % 2u != 0u) {
    std::fprintf(stderr, "batch balanced sample count invalid: %zu\n", samples);
    return false;
  }
  std::vector<std::int32_t> input(count, 5);
  auto device = rund::compute::open(TargetFor(backend));
  if (!device) {
    std::fprintf(stderr, "batch %s open failed: %.*s\n", Name(backend),
                 static_cast<int>(device.error().size()),
                 device.error().data());
    return false;
  }
  auto program = rund::compute::on(*device)
                     .map<std::int32_t>("batch-map", count,
                                        [](auto value) {
                                          auto first = value * 3 + 7;
                                          return first * 5 - value;
                                        })
                     .compile();
  if (!program) {
    std::fprintf(stderr, "batch %s compile failed: %.*s\n", Name(backend),
                 static_cast<int>(program.error().size()),
                 program.error().data());
    return false;
  }

  using Job = rund::compute::Job<std::int32_t(std::int32_t)>;
  std::vector<Job> jobs{};
  jobs.reserve(JobCount);
  rund::compute::Batch batch{};
  for (std::size_t index = 0u; index < JobCount; ++index) {
    auto job = program->resident(input);
    if (!job || !job->run()) {
      std::fprintf(stderr, "batch %s resident/warm failed at %zu\n",
                   Name(backend), index);
      return false;
    }
    jobs.push_back(std::move(*job));
    if (!batch.add(jobs.back())) {
      std::fprintf(stderr, "batch %s admission failed at %zu\n", Name(backend),
                   index);
      return false;
    }
  }
  if (!batch.run()) {
    std::fprintf(stderr, "batch %s warmup failed\n", Name(backend));
    return false;
  }

  std::vector<double> serial_wall{};
  std::vector<double> serial_wait{};
  std::vector<double> serial_kernel{};
  std::vector<double> serial_residual{};
  std::vector<double> batch_wall{};
  std::vector<double> batch_wait{};
  std::vector<double> batch_kernel{};
  std::vector<double> batch_residual{};
  std::vector<double> paired_speedup{};
  for (std::vector<double> *const values :
       {&serial_wall, &serial_wait, &serial_kernel, &serial_residual,
        &batch_wall, &batch_wait, &batch_kernel, &batch_residual,
        &paired_speedup}) {
    values->reserve(samples);
  }

  WarmCounters warm{};
  std::uint64_t serial_submits = 0u;
  std::uint64_t serial_dispatches = 0u;
  std::uint64_t batch_submits = 0u;
  std::uint64_t job_submits = 0u;
  std::uint64_t batch_dispatches = 0u;
  const auto measure_serial = [&]() {
    std::uint64_t submits = 0u;
    std::uint64_t dispatches = 0u;
    std::uint64_t wait_ns = 0u;
    std::uint64_t kernel_ns = 0u;
    const auto begin = Clock::now();
    for (Job &job : jobs) {
      const auto status = job.run();
      if (!status) {
        std::fprintf(stderr, "batch %s serial run failed: %.*s\n",
                     Name(backend), static_cast<int>(status.error().size()),
                     status.error().data());
        return false;
      }
      const auto stats = job.stats();
      warm.observe(stats);
      ::rund::detail::counter::Accumulate(submits, stats.command_submits);
      ::rund::detail::counter::Accumulate(dispatches, stats.dispatches);
      ::rund::detail::counter::Accumulate(wait_ns, stats.submit_wait_ns);
      ::rund::detail::counter::Accumulate(kernel_ns, stats.kernel_ns);
    }
    const auto end = Clock::now();
    const double wall =
        std::chrono::duration<double, std::micro>(end - begin).count();
    const double wait = static_cast<double>(wait_ns) / 1'000.0;
    serial_wall.push_back(wall);
    serial_wait.push_back(wait);
    serial_kernel.push_back(static_cast<double>(kernel_ns) / 1'000.0);
    serial_residual.push_back(std::max(wall - wait, 0.0));
    serial_submits = submits;
    serial_dispatches = dispatches;
    if (submits != JobCount) {
      std::fprintf(stderr, "batch %s serial submit contract failed: %llu/%zu\n",
                   Name(backend), static_cast<unsigned long long>(submits),
                   JobCount);
      return false;
    }
    return true;
  };
  const auto measure_batch = [&]() {
    const auto begin = Clock::now();
    const auto status = batch.run();
    const auto end = Clock::now();
    if (!status) {
      std::fprintf(stderr, "batch %s run failed: %.*s\n", Name(backend),
                   static_cast<int>(status.error().size()),
                   status.error().data());
      return false;
    }
    const auto stats = batch.stats();
    warm.observe(stats);
    job_submits = 0u;
    bool job_shared_zero = true;
    for (Job &job : jobs) {
      const auto local = job.stats();
      warm.observe(local);
      ::rund::detail::counter::Accumulate(job_submits, local.command_submits);
      job_shared_zero = job_shared_zero && BatchJobSharedZero(local);
    }
    if (stats.command_submits != 1u || !job_shared_zero) {
      std::fprintf(
          stderr,
          "batch %s shared submit contract failed: batch=%llu jobs=%llu\n",
          Name(backend), static_cast<unsigned long long>(stats.command_submits),
          static_cast<unsigned long long>(job_submits));
      return false;
    }
    const double wall =
        std::chrono::duration<double, std::micro>(end - begin).count();
    const double wait = static_cast<double>(stats.submit_wait_ns) / 1'000.0;
    batch_wall.push_back(wall);
    batch_wait.push_back(wait);
    batch_kernel.push_back(static_cast<double>(stats.kernel_ns) / 1'000.0);
    batch_residual.push_back(std::max(wall - wait, 0.0));
    batch_submits = stats.command_submits;
    batch_dispatches = stats.dispatches;
    return true;
  };

  std::size_t serial_first = 0u;
  std::size_t batch_first = 0u;
  for (std::size_t sample = 0u; sample < samples; ++sample) {
    const bool serial_then_batch = sample % 2u == 0u;
    const bool pair_ok = serial_then_batch
                             ? measure_serial() && measure_batch()
                             : measure_batch() && measure_serial();
    if (!pair_ok) {
      return false;
    }
    serial_first += serial_then_batch ? 1u : 0u;
    batch_first += serial_then_batch ? 0u : 1u;
    const double batch_sample = batch_wall.back();
    paired_speedup.push_back(
        batch_sample == 0.0 ? 0.0 : serial_wall.back() / batch_sample);
  }

  // Hash and value oracles run only after every timed pair. The same retained
  // Job set is executed once serially and once as a Batch so readback cannot
  // heat, throttle, or otherwise order either measured path.
  std::array<HashEvidence, JobCount> serial_hashes{};
  for (Job &job : jobs) {
    if (!job.run()) {
      std::fprintf(stderr, "batch %s serial oracle run failed\n",
                   Name(backend));
      return false;
    }
  }
  for (std::size_t index = 0u; index < jobs.size(); ++index) {
    auto output = jobs[index].read();
    if (!output || output->size() != count ||
        !std::all_of(output->begin(), output->end(),
                     [](const std::int32_t value) { return value == 105; })) {
      std::fprintf(stderr, "batch %s serial output failed at %zu\n",
                   Name(backend), index);
      return false;
    }
    const auto stats = jobs[index].stats();
    serial_hashes[index] =
        HashEvidence{.graph = stats.graph_hash, .output = stats.output_hash};
    if (!serial_hashes[index].valid()) {
      std::fprintf(stderr, "batch %s serial hash missing at %zu\n",
                   Name(backend), index);
      return false;
    }
  }
  if (!batch.run()) {
    std::fprintf(stderr, "batch %s oracle run failed\n", Name(backend));
    return false;
  }
  for (Job &job : jobs) {
    if (!BatchJobSharedZero(job.stats())) {
      std::fprintf(stderr, "batch %s oracle evidence duplicated\n",
                   Name(backend));
      return false;
    }
  }

  HashEvidence evidence{};
  bool parity = true;
  for (std::size_t index = 0u; index < jobs.size(); ++index) {
    auto output = jobs[index].read();
    if (!output || output->size() != count ||
        !std::all_of(output->begin(), output->end(),
                     [](const std::int32_t value) { return value == 105; })) {
      std::fprintf(stderr, "batch %s output failed at %zu\n", Name(backend),
                   index);
      return false;
    }
    const auto stats = jobs[index].stats();
    const HashEvidence current{.graph = stats.graph_hash,
                               .output = stats.output_hash};
    if (index == 0u) {
      evidence = current;
    }
    parity = parity && current.valid() && current == evidence &&
             current == serial_hashes[index];
  }
  if (!batch_reference.valid()) {
    batch_reference = evidence;
  } else {
    parity = parity && evidence == batch_reference;
  }

  const double serial_wall_us = Median(serial_wall);
  const double batch_wall_us = Median(batch_wall);
  const double serial_jobs_per_s =
      serial_wall_us == 0.0
          ? 0.0
          : static_cast<double>(JobCount) * 1'000'000.0 / serial_wall_us;
  const double batch_jobs_per_s =
      batch_wall_us == 0.0
          ? 0.0
          : static_cast<double>(JobCount) * 1'000'000.0 / batch_wall_us;
  const double speedup =
      batch_wall_us == 0.0 ? 0.0 : serial_wall_us / batch_wall_us;
  const bool balanced =
      serial_first == batch_first && serial_first + batch_first == samples;
  const bool contract = parity && warm.zero() && serial_submits == JobCount &&
                        batch_submits == 1u && job_submits == 0u &&
                        serial_dispatches == batch_dispatches && balanced;
  std::printf(
      "batch,%s,%s,%zu,%zu,%zu,%zu,%zu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
      "%.3f,%.3f,%.3f,%.3f,%.6f,%.6f,%llu,%llu,%llu,%llu,%llu,%llu,%llu,"
      "%u,%u\n",
      Name(backend), contract ? "ok" : "contract_failed", JobCount, count,
      samples, serial_first, batch_first, serial_wall_us, batch_wall_us,
      Median(serial_wait), Median(batch_wait), Median(serial_kernel),
      Median(batch_kernel), Median(serial_residual), Median(batch_residual),
      serial_jobs_per_s, batch_jobs_per_s, speedup, Median(paired_speedup),
      static_cast<unsigned long long>(serial_submits),
      static_cast<unsigned long long>(batch_submits),
      static_cast<unsigned long long>(job_submits),
      static_cast<unsigned long long>(serial_dispatches),
      static_cast<unsigned long long>(batch_dispatches),
      static_cast<unsigned long long>(evidence.graph),
      static_cast<unsigned long long>(evidence.output), parity ? 1u : 0u,
      warm.zero() ? 1u : 0u);
  return contract;
}

} // namespace rund::measure::compute
