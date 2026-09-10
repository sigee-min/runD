#pragma once

#include "capture.hpp"
#include "output.hpp"

#include <chrono>
#include <cstdio>
#include <vector>

namespace rund::measure::compute {

// Concrete suites own program construction and output predicates. This
// generic boundary owns only the common prepared run and CSV projection.
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
