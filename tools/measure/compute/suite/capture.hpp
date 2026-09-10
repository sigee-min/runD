#pragma once

#include "reference.hpp"

#include <cstdio>

namespace rund::measure::compute {

// Resident Jobs expose either one output or a grouped read_all() projection.
// This helper validates that caller-owned projection and records only the
// canonical graph/output evidence shared by the measurement suites.
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

} // namespace rund::measure::compute
