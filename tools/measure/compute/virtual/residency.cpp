#include "residency.hpp"

#include "../suite/core.hpp"
#include "backing.hpp"
#include "model.hpp"
#include "oracle.hpp"
#include "product_route.hpp"
#include "report.hpp"
#include "residency/backing.hpp"

#include <accel/api.hpp>
#include <node/accel/pick.hpp>
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

[[nodiscard]] const char *RawOpenFailure(const Backend backend) noexcept {
  rund::AccelPolicy policy{};
  switch (backend) {
  case Backend::Metal:
    policy.preferred[0] = rund::AccelApi::Metal;
    break;
  case Backend::Vulkan:
    policy.preferred[0] = rund::AccelApi::Vulkan;
    break;
  case Backend::Cpu:
  case Backend::Unavailable:
    return "not_accelerator";
  }
  policy.preferred_count = 1u;
  policy.allow_fake = false;
  const rund::AccelDevice probed = rund::node::accel::PickAccel(policy);
  return probed.check.reason == nullptr ? "reason_unavailable"
                                        : probed.check.reason;
}

[[nodiscard]] std::uint64_t
ActivePages(const std::size_t active_count) noexcept {
  return active_count / PageElements +
         static_cast<std::uint64_t>(active_count % PageElements != 0u);
}

} // namespace
} // namespace virtual_residency

void PrintVirtualResidencyColumns() { virtual_residency::PrintColumns(); }

bool MeasureVirtualResidency(const Backend backend,
                             const VirtualResidencyBacking backing_mode) {
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
    std::printf("environment,%s,open_failed,%u,", Name(backend),
                static_cast<unsigned>(device.code()));
    PrintCsv(device.error());
    std::fputs(",\"\",\"\",\"\"\n", stdout);
    std::fprintf(stderr, "virtual residency %s open failed: %.*s raw=%s\n",
                 Name(backend), static_cast<int>(device.error().size()),
                 device.error().data(), RawOpenFailure(backend));
    if (backend != Backend::Cpu) {
      (void)ReportEnvironment(Backend::Cpu);
    }
    return false;
  }
  if (!ReportEnvironment(backend, *device) ||
      (backend != Backend::Cpu && !ReportEnvironment(Backend::Cpu))) {
    return false;
  }
  ProductRouteObserver product_route{*device};
  std::vector<std::int32_t> seeded(LogicalElements);
  std::vector<std::int32_t> observed(LogicalElements);
  std::vector<std::byte> poisoned(LogicalElements * sizeof(std::int32_t),
                                  TailPoison);
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
  const ResidencyBackings backings =
      CreateResidencyBackings(*device, backing_mode, LogicalElements);
  const auto initialized_output =
      backings ? backings.output->write(0u, poisoned)
               : ::rund::compute::Status::fail(
                     ::rund::compute::Reason::BufferCapacity);
  auto input = virtual_buffer<std::int32_t>(LogicalElements, backings.input);
  auto output = virtual_buffer<std::int32_t>(LogicalElements, backings.output);
  auto prepared =
      program && input && output
          ? virtual_pipeline(
                *program, *input, *output,
                ResidencyConfig{.device_resident_bytes = ResidentBytes,
                                .host_resident_bytes = HostResidentBytes})
          : decltype(virtual_pipeline(
                *program, *input, *output,
                ResidencyConfig{.device_resident_bytes = ResidentBytes,
                                .host_resident_bytes = HostResidentBytes}))::
                fail(::rund::compute::Reason::PipelineInvalid);
  const auto prepared_at = Clock::now();
  if (!program || !backings || !initialized_output || !input || !output ||
      !prepared) {
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
  const auto seeded_status = backings.input->write(0u, seeded_bytes);
  const auto seeded_at = Clock::now();
  const auto plan = prepared->plan();
  if (!seeded_status) {
    std::fprintf(stderr, "virtual residency %s seed failed\n", Name(backend));
    return false;
  }
  product_route.reset();
  const auto cold_status = prepared->run();
  const auto ran_at = Clock::now();
  const ProductRouteEvidence cold_route = product_route.evidence();
  const auto cold_read =
      cold_status ? backings.output->read(0u, observed_bytes) : cold_status;
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
      !ExactProductRoute(cold_route, backend, PageCount,
                         LogicalElements * sizeof(std::int32_t), 1u,
                         backings.resident) ||
      !ExactProfile(
          *cold_profile, plan, backend, LogicalElements, 0u,
          ExactGpuDrivenProductRoute(cold_route, backend, PageCount,
                                     LogicalElements * sizeof(std::int32_t), 1u,
                                     backings.resident),
          backings.resident)) {
    if (cold_profile) {
      const auto &facts = cold_profile->execution();
      const auto &residency = facts.pipeline.residency;
      std::fprintf(
          stderr,
          "virtual residency cold facts pages=%llu epochs=%llu frames=%llu "
          "loads=%llu hits=%llu late=%llu prefetch=%llu out=%llu "
          "backing=%llu/%llu transfer=%llu/%llu dispatch=%llu hash=%llu/%llu\n",
          static_cast<unsigned long long>(residency.page_count),
          static_cast<unsigned long long>(residency.epoch_count),
          static_cast<unsigned long long>(residency.resident_frames_peak),
          static_cast<unsigned long long>(residency.page_in_count),
          static_cast<unsigned long long>(residency.cache_hit_count),
          static_cast<unsigned long long>(residency.late_page_count),
          static_cast<unsigned long long>(residency.prefetch_count),
          static_cast<unsigned long long>(residency.page_out_count),
          static_cast<unsigned long long>(residency.backing_read_bytes),
          static_cast<unsigned long long>(residency.backing_write_bytes),
          static_cast<unsigned long long>(facts.uploaded_bytes),
          static_cast<unsigned long long>(facts.downloaded_bytes),
          static_cast<unsigned long long>(facts.dispatches),
          static_cast<unsigned long long>(facts.output_hash),
          static_cast<unsigned long long>(
              ContentHash(observed, LogicalElements)));
    }
    std::fprintf(stderr, "virtual residency %s full cold evidence failed\n",
                 Name(backend));
    return false;
  }
  const WallEvidence cold_wall =
      ColdWall(cold_begin, authored_at, compiled_at, prepared_at, seeded_at,
               ran_at, read_at);
  PrintEvidence(
      "cold", backend, LogicalElements, cold_wall,
      std::bit_cast<std::uint32_t>(observed.front()), *cold_profile, plan,
      *preparation, cold_observation, cold_route,
      ProductRouteStatus(cold_route, backend, PageCount, backings.resident),
      1u);
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
    const auto reset = backings.output->write(0u, poisoned);
    const auto invalidated = reset ? backings.output->invalidate() : reset;
    const auto conditioned =
        invalidated ? prepared->run(active_count) : invalidated;
    if (!conditioned || !prepared->begin_samples()) {
      std::fprintf(stderr,
                   "virtual residency %s active=%zu conditioning/sample "
                   "failed\n",
                   Name(backend), active_count);
      return false;
    }
    product_route.reset();
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
    const ProductRouteEvidence warm_route = product_route.evidence();
    const auto terminal_read = backings.output->read(0u, observed_bytes);
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
        !ExactProductRoute(warm_route, backend, ActivePages(active_count),
                           active_count * sizeof(std::int32_t), WarmSamples,
                           backings.resident) ||
        !ExactProfile(*warm_profile, plan, backend, active_count, WarmSamples,
                      ExactGpuDrivenProductRoute(
                          warm_route, backend, ActivePages(active_count),
                          active_count * sizeof(std::int32_t), WarmSamples,
                          backings.resident),
                      backings.resident)) {
      if (warm_profile) {
        const auto &facts = warm_profile->execution();
        const auto &residency = facts.pipeline.residency;
        std::fprintf(
            stderr,
            "virtual residency facts active=%zu pages=%llu epochs=%llu "
            "loads=%llu hits=%llu late=%llu prefetch=%llu out=%llu "
            "backing=%llu/%llu transfer=%llu/%llu dispatch=%llu "
            "samples=%u/%u\n",
            active_count, static_cast<unsigned long long>(residency.page_count),
            static_cast<unsigned long long>(residency.epoch_count),
            static_cast<unsigned long long>(residency.page_in_count),
            static_cast<unsigned long long>(residency.cache_hit_count),
            static_cast<unsigned long long>(residency.late_page_count),
            static_cast<unsigned long long>(residency.prefetch_count),
            static_cast<unsigned long long>(residency.page_out_count),
            static_cast<unsigned long long>(residency.backing_read_bytes),
            static_cast<unsigned long long>(residency.backing_write_bytes),
            static_cast<unsigned long long>(facts.uploaded_bytes),
            static_cast<unsigned long long>(facts.downloaded_bytes),
            static_cast<unsigned long long>(facts.dispatches),
            residency.sampled_runs, residency.allocation_free_runs);
      }
      std::fprintf(stderr,
                   "virtual residency %s active=%zu warm evidence failed\n",
                   Name(backend), active_count);
      return false;
    }
    const WallEvidence warm_wall = WarmWall(samples, active_count);
    PrintEvidence(
        phase_names[phase], backend, active_count, warm_wall,
        active_count == 0u ? 0u
                           : std::bit_cast<std::uint32_t>(observed.front()),
        *warm_profile, plan, *preparation, observation, warm_route,
        ProductRouteStatus(warm_route, backend, ActivePages(active_count),
                           backings.resident),
        WarmSamples);
  }
  return true;
}

} // namespace rund::measure::compute
