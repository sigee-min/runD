#include "residency.hpp"

#include "backing.hpp"
#include "model.hpp"
#include "oracle.hpp"
#include "report.hpp"

#include <rund/compute/virtual.hpp>

#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace rund::measure::compute {
namespace virtual_residency {
namespace {

[[nodiscard]] double microseconds(const Clock::duration duration) noexcept {
  return std::chrono::duration<double, std::micro>(duration).count();
}

[[nodiscard]] WallEvidence
ColdWall(const Clock::time_point begin, const Clock::time_point authored,
         const Clock::time_point compiled, const Clock::time_point prepared,
         const Clock::time_point seeded, const Clock::time_point ran,
         const Clock::time_point read) noexcept {
  const double first_result = microseconds(read - begin);
  return {
      .first_result_us = first_result,
      .logical_elements_per_s = first_result == 0.0
                                    ? 0.0
                                    : static_cast<double>(LogicalElements) *
                                          1'000'000.0 / first_result,
      .author_us = microseconds(authored - begin),
      .compile_us = microseconds(compiled - authored),
      .prepare_us = microseconds(prepared - compiled),
      .seed_us = microseconds(seeded - prepared),
      .run_us = microseconds(ran - seeded),
      .read_us = microseconds(read - ran),
  };
}

[[nodiscard]] WallEvidence WarmWall(WallSamples &samples,
                                    const std::size_t active_count) noexcept {
  samples.sort();
  const double p50 = samples.p50();
  return {
      .warm_p50_us = p50,
      .warm_p95_us = samples.p95(),
      .logical_elements_per_s =
          p50 == 0.0 ? 0.0
                     : static_cast<double>(active_count) * 1'000'000.0 / p50,
  };
}

} // namespace
} // namespace virtual_residency

void PrintVirtualResidencyColumns() { virtual_residency::PrintColumns(); }

bool MeasureVirtualResidency(const Backend backend) {
  using namespace virtual_residency;
  using ::rund::compute::ResidencyConfig;
  using ::rund::compute::virtual_buffer;
  using ::rund::compute::virtual_pipeline;
  if (backend == Backend::Unavailable) {
    std::fputs("virtual residency backend invalid\n", stderr);
    return false;
  }

  auto device = ::rund::compute::open(TargetFor(backend));
  if (!device) {
    std::fprintf(stderr, "virtual residency %s open failed: %.*s\n",
                 Name(backend), static_cast<int>(device.error().size()),
                 device.error().data());
    return false;
  }
  std::vector<std::int32_t> seeded(LogicalElements);
  std::vector<std::int32_t> observed(LogicalElements);
  Seed(seeded);
  const auto seeded_bytes = std::as_bytes(std::span{seeded});
  auto observed_bytes = std::as_writable_bytes(std::span{observed});

  const auto cold_begin = Clock::now();
  auto authored = ::rund::compute::on(*device).map<std::int32_t>(
      "measure-virtual-residency", PageElements,
      [](auto value) { return (value + 5) * 3; });
  const auto authored_at = Clock::now();
  auto program = std::move(authored).compile();
  const auto compiled_at = Clock::now();
  auto input_backing =
      std::make_shared<MemoryBacking>(LogicalElements * sizeof(std::int32_t));
  auto output_backing =
      std::make_shared<MemoryBacking>(LogicalElements * sizeof(std::int32_t));
  auto input = virtual_buffer<std::int32_t>(LogicalElements, input_backing);
  auto output = virtual_buffer<std::int32_t>(LogicalElements, output_backing);
  auto prepared =
      program && input && output
          ? virtual_pipeline(*program, *input, *output,
                             ResidencyConfig{.slots = SlotCapacity})
          : decltype(virtual_pipeline(*program, *input, *output,
                                      ResidencyConfig{.slots = SlotCapacity}))::
                fail(::rund::compute::Reason::PipelineInvalid);
  const auto prepared_at = Clock::now();
  if (!program || !input || !output || !prepared) {
    if (prepared.reason() == ::rund::compute::Reason::BackendUnsupported) {
      std::fprintf(stderr,
                   "virtual residency status=blocked backend=%s "
                   "reason=compute_backend_unsupported\n",
                   Name(backend));
      return false;
    }
    std::fprintf(stderr, "virtual residency %s cold prepare failed\n",
                 Name(backend));
    return false;
  }
  auto preparation = prepared->profile();
  if (!preparation || !ExactPreparationEvidence(*preparation)) {
    std::fprintf(stderr, "virtual residency %s preparation evidence failed\n",
                 Name(backend));
    return false;
  }
  const auto seeded_status = input_backing->write(0u, seeded_bytes);
  const auto seeded_at = Clock::now();
  const auto plan = prepared->plan();
  if (!seeded_status) {
    std::fprintf(stderr, "virtual residency %s seed failed\n", Name(backend));
    return false;
  }
  const auto cold_status = prepared->run();
  const auto ran_at = Clock::now();
  const auto cold_read =
      cold_status ? output_backing->read(0u, observed_bytes) : cold_status;
  const auto read_at = Clock::now();
  ObservationEvidence cold_observation{};
  if (cold_read) {
    ++cold_observation.terminal_reads;
  }
  auto cold_profile = prepared->profile();
  if (cold_profile) {
    ++cold_observation.profile_projections;
  }
  if (!cold_read || !cold_profile || !ValidOutput(observed, LogicalElements) ||
      !cold_observation.exact_terminal_only() ||
      cold_profile->execution().output_hash !=
          ContentHash(observed, LogicalElements) ||
      !ExactProfile(*cold_profile, plan, backend, LogicalElements, 0u)) {
    std::fprintf(stderr, "virtual residency %s full cold evidence failed\n",
                 Name(backend));
    return false;
  }
  const WallEvidence cold_wall =
      ColdWall(cold_begin, authored_at, compiled_at, prepared_at, seeded_at,
               ran_at, read_at);
  PrintEvidence("cold", backend, LogicalElements, cold_wall,
                std::bit_cast<std::uint32_t>(observed.front()), *cold_profile,
                plan, *preparation, cold_observation);
  std::fprintf(stderr,
               "virtual residency %s cold phases us author=%.3f compile=%.3f "
               "prepare=%.3f seed=%.3f run=%.3f read=%.3f\n",
               Name(backend), cold_wall.author_us, cold_wall.compile_us,
               cold_wall.prepare_us, cold_wall.seed_us, cold_wall.run_us,
               cold_wall.read_us);

  constexpr std::array<const char *, ActiveCounts.size()> phase_names{
      "n0", "small", "mid", "capacity"};
  for (std::size_t phase = 0u; phase < ActiveCounts.size(); ++phase) {
    const std::size_t active_count = ActiveCounts[phase];
    output_backing->reset(TailPoison);
    const auto conditioned = prepared->run(active_count);
    if (!conditioned || !prepared->begin_samples()) {
      std::fprintf(stderr,
                   "virtual residency %s active=%zu conditioning/sample "
                   "failed\n",
                   Name(backend), active_count);
      return false;
    }
    WallSamples samples{};
    bool warm_ok = true;
    for (std::size_t sample = 0u; sample < WarmSamples; ++sample) {
      const auto begin = Clock::now();
      const auto status = prepared->run(active_count);
      const auto end = Clock::now();
      samples.microseconds[sample] = microseconds(end - begin);
      if (!status) {
        warm_ok = false;
        break;
      }
    }
    const auto samples_ended = prepared->end_samples();
    const auto terminal_read = output_backing->read(0u, observed_bytes);
    ObservationEvidence observation{};
    if (terminal_read) {
      ++observation.terminal_reads;
    }
    auto warm_profile = prepared->profile();
    if (warm_profile) {
      ++observation.profile_projections;
    }
    if (!warm_ok || !samples_ended || !terminal_read || !warm_profile ||
        !observation.exact_terminal_only() || prepared->plan() != plan ||
        !ValidOutput(observed, active_count) ||
        warm_profile->execution().output_hash !=
            ContentHash(observed, active_count) ||
        !ExactProfile(*warm_profile, plan, backend, active_count,
                      WarmSamples)) {
      std::fprintf(stderr,
                   "virtual residency %s active=%zu warm evidence failed\n",
                   Name(backend), active_count);
      return false;
    }
    const WallEvidence warm_wall = WarmWall(samples, active_count);
    PrintEvidence(phase_names[phase], backend, active_count, warm_wall,
                  active_count == 0u
                      ? 0u
                      : std::bit_cast<std::uint32_t>(observed.front()),
                  *warm_profile, plan, *preparation, observation);
  }
  return true;
}

} // namespace rund::measure::compute
