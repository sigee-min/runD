#include "core.hpp"

namespace rund::measure::compute {

#if !defined(RUND_COMPUTE_FOCUS)
bool Map(const char *label, const rund::compute::Target target,
         const std::size_t count, const std::size_t iterations) {
  std::vector<std::int32_t> input(count);
  for (std::size_t i = 0; i < count; ++i) {
    input[i] = static_cast<std::int32_t>(i * 17u + 3u);
  }
  const auto compile_begin = Clock::now();
  auto program =
      rund::compute::on(target)
          .template map<std::int32_t>("map-scale", count,
                                      [](auto value) {
                                        auto first = value * 3 + 7;
                                        auto second = first * 5 - value * 2;
                                        auto third = second * 7 + first * 3;
                                        return third * 11 - second;
                                      })
          .compile();
  const auto compile_end = Clock::now();
  const double compile_us =
      std::chrono::duration<double, std::micro>(compile_end - compile_begin)
          .count();
  std::printf("cold,%s,%zu,%.3f\n", label, count, compile_us);
  return Bench(label, Backend::Cpu, program, iterations,
               ReferenceKey{"map", "scale", count}, input);
}

bool NodeMap(const std::uint32_t workers, const std::size_t count,
             const std::size_t iterations) {
  std::vector<std::int32_t> input(count, 3);
  auto program =
      rund::compute::on(rund::compute::Target::cpu(workers))
          .map<std::int32_t>("node-map", count,
                             [](auto value) { return value * 3 + 7; })
          .compile();
  if (!program) {
    return false;
  }
  auto job = program->resident(input);
  if (!job) {
    return false;
  }
  std::vector<double> standalone_samples;
  standalone_samples.reserve(iterations);
  if (!job->run()) {
    return false;
  }
  WarmCounters standalone_counters{};
  for (std::size_t i = 0; i < iterations; ++i) {
    const auto begin = Clock::now();
    const auto result = job->run();
    const auto end = Clock::now();
    if (!result) {
      return false;
    }
    standalone_counters.observe(job->stats());
    standalone_samples.push_back(
        std::chrono::duration<double, std::micro>(end - begin).count());
  }
  HashEvidence standalone_evidence{};
  if (!CaptureOutput(*job, Backend::Cpu, "standalone_map",
                     standalone_evidence)) {
    return false;
  }
  const auto standalone_stats = job->stats();
  const auto standalone_memory = job->memory();
  const bool standalone_zero =
      ReportHostMap("standalone", count, Median(standalone_samples),
                    standalone_stats, standalone_memory, standalone_counters);
  std::vector<double> samples;
  samples.reserve(iterations);
  WarmCounters node_counters{};
  const auto run = rund::run(
      {.workers = workers, .scheduler = {.task_workers = workers}},
      [&](rund::Session &session) {
        auto warm = session.compute(*job).submit().wait();
        if (!warm)
          return;
        for (std::size_t i = 0; i < iterations; ++i) {
          const auto begin = Clock::now();
          auto result = session.compute(*job).submit().wait();
          const auto end = Clock::now();
          if (!result)
            return;
          node_counters.observe(job->stats());
          samples.push_back(
              std::chrono::duration<double, std::micro>(end - begin).count());
        }
      });
  if (!run || samples.size() != iterations) {
    return false;
  }
  HashEvidence node_evidence{};
  if (!CaptureOutput(*job, Backend::Cpu, "node_map", node_evidence)) {
    return false;
  }
  const bool parity = standalone_evidence == node_evidence;
  if (!parity) {
    std::fprintf(
        stderr, "host map parity failed: standalone=%llu/%llu node=%llu/%llu\n",
        static_cast<unsigned long long>(standalone_evidence.graph),
        static_cast<unsigned long long>(standalone_evidence.output),
        static_cast<unsigned long long>(node_evidence.graph),
        static_cast<unsigned long long>(node_evidence.output));
  }
  const auto stats = job->stats();
  const auto memory = job->memory();
  const bool node_zero = ReportHostMap("node", count, Median(samples), stats,
                                       memory, node_counters);
  return standalone_zero && node_zero && parity;
}

bool NodeOrchestrationOn(const rund::compute::Target target,
                         const Backend backend, const std::uint32_t workers,
                         const std::size_t count,
                         const std::size_t iterations) {
  std::vector<std::int32_t> input(count, 3);
  auto program =
      rund::compute::on(target)
          .template map<std::int32_t>("node-orchestration", count,
                                      [](auto value) { return value * 3 + 7; })
          .compile();
  if (!program) {
    std::fprintf(stderr, "orchestration %s compile failed: %.*s\n",
                 Name(backend), static_cast<int>(program.error().size()),
                 program.error().data());
    return false;
  }
  auto job = program->resident(input);
  if (!job) {
    std::fprintf(stderr, "orchestration %s resident failed: %.*s\n",
                 Name(backend), static_cast<int>(job.error().size()),
                 job.error().data());
    return false;
  }

  std::vector<double> submit_samples{};
  std::vector<double> peer_samples{};
  std::vector<double> completion_samples{};
  std::vector<double> total_samples{};
  submit_samples.reserve(iterations);
  peer_samples.reserve(iterations);
  completion_samples.reserve(iterations);
  total_samples.reserve(iterations);
  std::size_t completed = 0u;
  std::size_t peer_completed = 0u;
  WarmCounters warm{};
  const std::uint32_t task_workers = backend == Backend::Cpu ? workers : 1u;
  const auto run = rund::run(
      {.workers = workers,
       .scheduler = {.task_workers = task_workers,
                     .task_capacity = 64u,
                     .ready_queue_capacity = 64u}},
      [&](rund::Session &session) {
        if (!session.compute(*job).submit().wait()) {
          return;
        }
        for (std::size_t i = 0; i < iterations; ++i) {
          const auto begin = Clock::now();
          auto task = session.compute(*job).submit();
          const auto submitted = Clock::now();
          const auto peer_begin = Clock::now();
          const auto peer = rund::task::spawn("compute-measure-peer", [] {});
          const bool peer_ok = static_cast<bool>(rund::task::join(peer));
          const auto peer_end = Clock::now();
          const auto result = task.wait();
          const auto end = Clock::now();
          if (!result || !peer_ok) {
            return;
          }
          warm.observe(job->stats());
          ++completed;
          ++peer_completed;
          submit_samples.push_back(
              std::chrono::duration<double, std::micro>(submitted - begin)
                  .count());
          peer_samples.push_back(
              std::chrono::duration<double, std::micro>(peer_end - peer_begin)
                  .count());
          completion_samples.push_back(
              std::chrono::duration<double, std::micro>(end - submitted)
                  .count());
          total_samples.push_back(
              std::chrono::duration<double, std::micro>(end - begin).count());
        }
      });
  if (!run || completed != iterations || peer_completed != iterations) {
    const std::string_view error = run.error();
    std::fprintf(stderr,
                 "orchestration %s run failed: %.*s completed=%zu peer=%zu\n",
                 Name(backend), static_cast<int>(error.size()), error.data(),
                 completed, peer_completed);
    return false;
  }
  HashEvidence evidence{};
  if (!CaptureOutput(*job, backend, "orchestration", evidence)) {
    return false;
  }
  const bool reference_ok = CheckReference(
      backend, ReferenceKey{"orchestration", "map", count}, evidence);
  const auto stats = job->stats();
  std::printf(
      "orchestration,%s,%s,%.3f,%.3f,%.3f,%.3f,%zu,%llu,%llu,%llu,%llu,%llu,"
      "%u",
      Name(backend), reference_ok ? "ok" : "reference_failed",
      Median(submit_samples), Median(peer_samples), Median(completion_samples),
      Median(total_samples), peer_completed,
      static_cast<unsigned long long>(run.tasks().external_parks()),
      static_cast<unsigned long long>(run.tasks().external_wakes()),
      static_cast<unsigned long long>(run.tasks().parked()),
      static_cast<unsigned long long>(run.tasks().resumed()),
      static_cast<unsigned long long>(run.tasks().task_workers()),
      task_workers);
  PrintStats(stats);
  PrintWarm(warm);
  std::putchar('\n');
  return warm.zero() && reference_ok && run.tasks().external_parks() != 0u &&
         run.tasks().external_parks() == run.tasks().external_wakes();
}

bool NodeOrchestration(const Backend backend, const std::uint32_t workers,
                       const std::size_t count, const std::size_t iterations) {
  const auto target = backend == Backend::Cpu
                          ? rund::compute::Target::cpu(workers)
                          : TargetFor(backend);
  return NodeOrchestrationOn(target, backend, workers, count, iterations);
}

bool InflightVulkan(const std::uint32_t workers, const std::size_t count,
                    const std::size_t samples) {
  std::vector<std::int32_t> input(count, 5);
  auto program = rund::compute::on(rund::compute::Target::vulkan())
                     .map<std::int32_t>("inflight-map", count,
                                        [](auto value) {
                                          auto first = value * 3 + 7;
                                          auto second = first * 5 - value * 2;
                                          return second * 7 + first;
                                        })
                     .compile();
  if (!program) {
    std::fprintf(stderr, "inflight vulkan compile failed: %.*s\n",
                 static_cast<int>(program.error().size()),
                 program.error().data());
    return false;
  }

  using Job = rund::compute::Job<std::int32_t(std::int32_t)>;
  auto first = program->resident(input);
  if (!first || !first->run()) {
    std::fputs("inflight vulkan first resident/warm failed\n", stderr);
    return false;
  }
  const auto envelope = first->stats();
  const auto capacity = static_cast<std::size_t>(envelope.command_capacity);
  constexpr auto task_limit =
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max());
  if (capacity == 0u || capacity > task_limit / 4u) {
    std::fprintf(stderr, "inflight vulkan invalid command capacity: %zu\n",
                 capacity);
    return false;
  }

  std::vector<Job> jobs{};
  jobs.reserve(capacity);
  jobs.push_back(std::move(*first));
  for (std::size_t index = 1u; index < capacity; ++index) {
    auto job = program->resident(input);
    if (!job || !job->run()) {
      std::fprintf(stderr, "inflight vulkan resident/warm failed at %zu\n",
                   index);
      return false;
    }
    jobs.push_back(std::move(*job));
  }

  std::vector<double> serial_samples{};
  std::vector<double> concurrent_samples{};
  serial_samples.reserve(samples);
  concurrent_samples.reserve(samples);
  std::vector<HashEvidence> serial_hashes(capacity);
  std::vector<rund::compute::Submission> tasks(capacity);
  WarmCounters warm{};
  rund::compute::Stats batch{};
  std::size_t completed = 0u;
  const auto run = rund::run(
      {.workers = workers,
       .scheduler = {.task_workers = static_cast<std::uint32_t>(capacity),
                     .task_capacity = static_cast<std::uint32_t>(capacity * 4u),
                     .ready_queue_capacity =
                         static_cast<std::uint32_t>(capacity * 4u)}},
      [&](rund::Session &session) {
        for (std::size_t sample = 0u; sample < samples; ++sample) {
          const auto begin = Clock::now();
          for (Job &job : jobs) {
            if (!session.compute(job).submit().wait()) {
              return;
            }
            warm.observe(job.stats());
          }
          const auto end = Clock::now();
          serial_samples.push_back(
              std::chrono::duration<double, std::micro>(end - begin).count());
        }
        for (std::size_t index = 0u; index < jobs.size(); ++index) {
          if (!CaptureOutput(jobs[index], Backend::Vulkan, "inflight-serial",
                             serial_hashes[index])) {
            return;
          }
        }

        for (std::size_t sample = 0u; sample < samples; ++sample) {
          const auto begin = Clock::now();
          for (std::size_t index = 0u; index < jobs.size(); ++index) {
            tasks[index] = session.compute(jobs[index]).submit();
          }
          for (auto &task : tasks) {
            if (!task.wait()) {
              return;
            }
          }
          const auto end = Clock::now();
          rund::compute::Stats envelope_stats{};
          for (Job &job : jobs) {
            const rund::compute::Stats stats = job.stats();
            warm.observe(stats);
            AccumulateInflight(envelope_stats, stats);
          }
          batch = envelope_stats;
          concurrent_samples.push_back(
              std::chrono::duration<double, std::micro>(end - begin).count());
          ++completed;
        }
      });
  if (!run || completed != samples || serial_samples.size() != samples ||
      concurrent_samples.size() != samples) {
    std::fprintf(stderr, "inflight vulkan run failed: completed=%zu\n",
                 completed);
    return false;
  }

  HashEvidence evidence{};
  bool hash_parity = true;
  for (std::size_t index = 0u; index < jobs.size(); ++index) {
    HashEvidence current{};
    if (!CaptureOutput(jobs[index], Backend::Vulkan, "inflight-concurrent",
                       current)) {
      return false;
    }
    if (index == 0u) {
      evidence = current;
    }
    hash_parity =
        hash_parity && current == evidence && current == serial_hashes[index];
  }

  const double serial_median_us = Median(serial_samples);
  const double concurrent_median_us = Median(concurrent_samples);
  const double serial_jobs_per_s =
      static_cast<double>(capacity) * 1'000'000.0 / serial_median_us;
  const double concurrent_jobs_per_s =
      static_cast<double>(capacity) * 1'000'000.0 / concurrent_median_us;
  const double concurrent_items_per_s = static_cast<double>(capacity) *
                                        static_cast<double>(count) *
                                        1'000'000.0 / concurrent_median_us;
  const double speedup = serial_median_us / concurrent_median_us;
  const bool bounded = ValidInflight(
      static_cast<std::uint64_t>(capacity), batch.command_capacity,
      batch.command_inflight_peak, batch.command_capacity_rejections);
  const bool contract = bounded && hash_parity && warm.zero() &&
                        batch.command_submits >= capacity &&
                        batch.dispatches >= capacity;
  std::printf(
      "inflight,vulkan,%s,%zu,%zu,%zu,%.3f,%.3f,%.3f,%.3f,%.3f,%.6f,"
      "%llu,%llu,%llu,%llu,%llu,%u,%u,%llu,%llu\n",
      contract ? "ok" : "contract_failed", capacity, count, samples,
      serial_median_us, concurrent_median_us, serial_jobs_per_s,
      concurrent_jobs_per_s, concurrent_items_per_s, speedup,
      static_cast<unsigned long long>(batch.command_capacity),
      static_cast<unsigned long long>(batch.command_inflight_peak),
      static_cast<unsigned long long>(batch.command_capacity_rejections),
      static_cast<unsigned long long>(evidence.graph),
      static_cast<unsigned long long>(evidence.output), hash_parity ? 1u : 0u,
      warm.zero() ? 1u : 0u,
      static_cast<unsigned long long>(batch.command_submits),
      static_cast<unsigned long long>(batch.dispatches));
  return contract;
}

#endif


} // namespace rund::measure::compute
