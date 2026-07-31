#include "core.hpp"

namespace rund::measure::compute {

#if !defined(RUND_COMPUTE_FOCUS)
bool Mixed(const std::uint32_t workers, const std::size_t count,
           const std::size_t iterations) {
  std::vector<std::int32_t> input(count, 5);
  const auto build = [count](const rund::compute::Target target) {
    return rund::compute::on(target)
        .template map<std::int32_t>("mixed-map", count,
                                    [](auto value) { return value * 5 - 3; })
        .compile();
  };
  auto cpu_program = build(rund::compute::Target::cpu(workers));
  auto metal_program = build(rund::compute::Target::metal());
  auto vulkan_program = build(rund::compute::Target::vulkan());
  if (!cpu_program || !metal_program || !vulkan_program) {
    std::fprintf(stderr, "mixed compile failed\n");
    return false;
  }
  auto cpu = cpu_program->resident(input);
  auto metal = metal_program->resident(input);
  auto vulkan = vulkan_program->resident(input);
  if (!cpu || !metal || !vulkan) {
    std::fprintf(stderr, "mixed resident failed\n");
    return false;
  }

  std::vector<double> serial_samples{};
  std::vector<double> mixed_samples{};
  serial_samples.reserve(iterations);
  mixed_samples.reserve(iterations);
  std::size_t completed = 0u;
  WarmCounters warm{};
  const auto run = rund::run(
      {.workers = workers,
       .scheduler = {.task_workers = workers,
                     .task_capacity = 64u,
                     .ready_queue_capacity = 64u}},
      [&](rund::Session &session) {
        if (!session.compute(*cpu).submit().wait() ||
            !session.compute(*metal).submit().wait() ||
            !session.compute(*vulkan).submit().wait()) {
          return;
        }
        for (std::size_t i = 0; i < iterations; ++i) {
          const auto serial_begin = Clock::now();
          const bool serial_ok = session.compute(*cpu).submit().wait() &&
                                 session.compute(*metal).submit().wait() &&
                                 session.compute(*vulkan).submit().wait();
          const auto serial_end = Clock::now();
          if (!serial_ok) {
            return;
          }
          warm.observe(cpu->stats());
          warm.observe(metal->stats());
          warm.observe(vulkan->stats());

          const auto mixed_begin = Clock::now();
          auto cpu_task = session.compute(*cpu).submit();
          auto metal_task = session.compute(*metal).submit();
          auto vulkan_task = session.compute(*vulkan).submit();
          const bool mixed_ok =
              cpu_task.wait() && metal_task.wait() && vulkan_task.wait();
          const auto mixed_end = Clock::now();
          if (!mixed_ok) {
            return;
          }
          warm.observe(cpu->stats());
          warm.observe(metal->stats());
          warm.observe(vulkan->stats());
          ++completed;
          serial_samples.push_back(std::chrono::duration<double, std::micro>(
                                       serial_end - serial_begin)
                                       .count());
          mixed_samples.push_back(
              std::chrono::duration<double, std::micro>(mixed_end - mixed_begin)
                  .count());
        }
      });
  if (!run || completed != iterations) {
    const std::string_view error = run.error();
    std::fprintf(stderr, "mixed run failed: %.*s completed=%zu\n",
                 static_cast<int>(error.size()), error.data(), completed);
    return false;
  }
  HashEvidence cpu_evidence{};
  HashEvidence metal_evidence{};
  HashEvidence vulkan_evidence{};
  bool output_ok = true;
  output_ok =
      CaptureOutput(*cpu, Backend::Cpu, "mixed", cpu_evidence) && output_ok;
  output_ok = CaptureOutput(*metal, Backend::Metal, "mixed", metal_evidence) &&
              output_ok;
  output_ok =
      CaptureOutput(*vulkan, Backend::Vulkan, "mixed", vulkan_evidence) &&
      output_ok;
  if (!output_ok) {
    return false;
  }
  const bool parity =
      cpu_evidence == metal_evidence && cpu_evidence == vulkan_evidence;
  if (!parity) {
    std::fprintf(stderr,
                 "mixed hash parity failed: cpu=%llu/%llu metal=%llu/%llu "
                 "vulkan=%llu/%llu\n",
                 static_cast<unsigned long long>(cpu_evidence.graph),
                 static_cast<unsigned long long>(cpu_evidence.output),
                 static_cast<unsigned long long>(metal_evidence.graph),
                 static_cast<unsigned long long>(metal_evidence.output),
                 static_cast<unsigned long long>(vulkan_evidence.graph),
                 static_cast<unsigned long long>(vulkan_evidence.output));
  }
  const double serial_us = Median(serial_samples);
  const double mixed_us = Median(mixed_samples);
  const double serial_rate = serial_us == 0.0 ? 0.0 : 3.0e6 / serial_us;
  const double mixed_rate = mixed_us == 0.0 ? 0.0 : 3.0e6 / mixed_us;
  std::printf(
      "mixed,cpu+metal+vulkan,%s,%.3f,%.3f,%.3f,%.3f,%llu,%llu,"
      "%llu,%llu,%llu,%llu,%llu,%llu,%u,%llu,%llu,%llu,"
      "%llu,%u\n",
      parity ? "ok" : "hash_mismatch", serial_us, mixed_us, serial_rate,
      mixed_rate, static_cast<unsigned long long>(run.tasks().external_parks()),
      static_cast<unsigned long long>(run.tasks().external_wakes()),
      static_cast<unsigned long long>(cpu_evidence.graph),
      static_cast<unsigned long long>(cpu_evidence.output),
      static_cast<unsigned long long>(metal_evidence.graph),
      static_cast<unsigned long long>(metal_evidence.output),
      static_cast<unsigned long long>(vulkan_evidence.graph),
      static_cast<unsigned long long>(vulkan_evidence.output), parity ? 1u : 0u,
      static_cast<unsigned long long>(warm.pipeline_compiles),
      static_cast<unsigned long long>(warm.buffer_allocations),
      static_cast<unsigned long long>(warm.download_events),
      static_cast<unsigned long long>(warm.uploaded_bytes),
      warm.zero() ? 1u : 0u);
  return warm.zero() && parity;
}

#endif

} // namespace rund::measure::compute
