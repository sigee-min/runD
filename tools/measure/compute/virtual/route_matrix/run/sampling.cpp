#include "../internal.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace rund::measure::compute::route_matrix {

RunResult run_one(Job &job, const CaseSpec &spec,
                  std::vector<std::int32_t> *observed, const bool read,
                  const bool reset) noexcept {
  if (!job.prepared || job.pipeline == nullptr || job.output == nullptr) {
    return {false, job.reason, 0.0};
  }
  if (reset && !fill_tail(job.output, elements(spec))) {
    return {false, ::rund::compute::Reason::BufferCapacity, 0.0};
  }
  const auto begin = Clock::now();
  const auto status = job.pipeline->run();
  const auto end = Clock::now();
  RunResult result{status.ok(), status.reason(), micros(end - begin)};
  if (!result.ok || !read) {
    return result;
  }
  if (observed == nullptr || observed->size() != elements(spec) ||
      !static_cast<bool>(
          job.output->read(0u, std::as_writable_bytes(std::span{*observed})))) {
    result.ok = false;
    result.reason = ::rund::compute::Reason::TransferInvalid;
  }
  return result;
}

bool read_output(const Job &job, const std::uint64_t count,
                 std::vector<std::int32_t> &observed) noexcept {
  if (!job.prepared || job.output == nullptr || count == 0u ||
      observed.size() != count) {
    return false;
  }
  return static_cast<bool>(
      job.output->read(0u, std::as_writable_bytes(std::span{observed})));
}

WarmSummary summarize(std::array<double, WarmSamples> &values,
                      const std::uint64_t count, const bool complete) noexcept {
  WarmSummary result{};
  result.complete = complete;
  if (!complete) {
    return result;
  }
  std::sort(values.begin(), values.end());
  const auto rank = [](const std::size_t percentile) constexpr {
    return (WarmSamples * percentile + 99u) / 100u - 1u;
  };
  result.p25_us = values[rank(25u)];
  result.p75_us = values[rank(75u)];
  result.p95_us = values[rank(95u)];
  result.p50_us = (values[29u] + values[30u]) / 2.0;
  std::array<double, WarmSamples> deviations{};
  for (std::size_t index = 0u; index < WarmSamples; ++index) {
    deviations[index] = std::abs(values[index] - result.p50_us);
  }
  std::sort(deviations.begin(), deviations.end());
  result.mad_us = (deviations[29u] + deviations[30u]) / 2.0;
  result.elements_per_s =
      result.p50_us <= 0.0
          ? 0.0
          : static_cast<double>(count) * 1'000'000.0 / result.p50_us;
  return result;
}

WarmCohort run_warm(Job &cpu, Job &selected, const CaseSpec &spec,
                    const std::uint64_t expected_count,
                    const std::uint64_t expected_hash,
                    std::vector<std::int32_t> &cpu_output,
                    std::vector<std::int32_t> &selected_output) noexcept {
  WarmCohort cohort{};
  const auto cpu_begin = cpu.pipeline->begin_samples();
  const auto selected_begin = selected.pipeline->begin_samples();
  bool samples_ok = cpu_begin.ok() && selected_begin.ok();
  auto sample_reason = !selected_begin.ok() ? selected_begin.reason()
                       : !cpu_begin.ok()    ? cpu_begin.reason()
                                            : ::rund::compute::Reason::Ok;
  for (std::size_t cycle = 0u; cycle < WarmSamples / 2u && samples_ok;
       ++cycle) {
    const auto sample = [&](Job &job, std::array<double, WarmSamples> &out,
                            const std::size_t index,
                            std::vector<std::int32_t> &check) noexcept {
      RunResult result = run_one(job, spec, nullptr, false, false);
      if (result.ok && (!read_output(job, expected_count, check) ||
                        !oracle::output_ok(check, spec, expected_hash))) {
        result.ok = false;
        result.reason = ::rund::compute::Reason::BackendFailed;
      }
      if (!result.ok && sample_reason == ::rund::compute::Reason::Ok) {
        sample_reason = result.reason;
      }
      out[index] = result.wall_us;
      return result;
    };
    const std::size_t first = cycle * 2u;
    const std::size_t second = first + 1u;
    const RunResult cpu_a = sample(cpu, cohort.cpu_values, first, cpu_output);
    const RunResult target_a =
        sample(selected, cohort.selected_values, first, selected_output);
    const RunResult target_b =
        sample(selected, cohort.selected_values, second, selected_output);
    const RunResult cpu_b = sample(cpu, cohort.cpu_values, second, cpu_output);
    samples_ok = cpu_a.ok && target_a.ok && target_b.ok && cpu_b.ok;
  }
  const auto cpu_end = cpu.pipeline->end_samples();
  const auto selected_end = selected.pipeline->end_samples();
  samples_ok = samples_ok && cpu_end.ok() && selected_end.ok();
  cohort.ok = samples_ok;
  cohort.reason = sample_reason != ::rund::compute::Reason::Ok ? sample_reason
                  : !selected_end.ok() ? selected_end.reason()
                  : !cpu_end.ok()      ? cpu_end.reason()
                                       : ::rund::compute::Reason::BackendFailed;
  return cohort;
}

} // namespace rund::measure::compute::route_matrix
