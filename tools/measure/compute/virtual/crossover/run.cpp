#include "internal.hpp"

#include <cstdio>
#include <utility>

namespace rund::measure::compute::virtual_crossover::detail {

bool run_abba(PreparedPoint &cpu, PreparedPoint &metal,
              const std::size_t active,
              virtual_residency::WallSamples *cpu_samples,
              virtual_residency::WallSamples *metal_samples,
              const bool timed) noexcept {
  for (std::size_t cycle = 0u; cycle < ConditioningRuns / 2u; ++cycle) {
    const auto run = [&](PreparedPoint &point,
                         virtual_residency::WallSamples *samples,
                         const std::size_t sample) noexcept {
      const auto begin = Clock::now();
      const auto status = point.pipeline.run(active);
      const auto end = Clock::now();
      if (timed && samples != nullptr) {
        samples->microseconds[sample] = microseconds(end - begin);
      }
      return static_cast<bool>(status);
    };
    const std::size_t first = cycle * 2u;
    const std::size_t second = first + 1u;
    if (!run(cpu, cpu_samples, first) || !run(metal, metal_samples, first) ||
        !run(metal, metal_samples, second) || !run(cpu, cpu_samples, second)) {
      return false;
    }
  }
  return true;
}

bool measure_point(const std::size_t logical_count, const std::size_t radius,
                   const ActiveRatio active_ratio,
                   const std::size_t ordinal) noexcept {
  const std::size_t active = active_count(logical_count, active_ratio);
  const auto cpu_prepare_begin = Clock::now();
  auto cpu = prepare(Backend::Cpu, logical_count, radius);
  const auto cpu_prepare_end = Clock::now();
  const auto metal_prepare_begin = Clock::now();
  auto metal = prepare(Backend::Metal, logical_count, radius);
  const auto metal_prepare_end = Clock::now();
  if (cpu == nullptr || metal == nullptr ||
      cpu->plan.residency.logical_bytes !=
          metal->plan.residency.logical_bytes ||
      cpu->plan.residency.page_bytes != metal->plan.residency.page_bytes ||
      cpu->plan.residency.page_count != metal->plan.residency.page_count ||
      cpu->plan.residency.frame_capacity !=
          metal->plan.residency.frame_capacity) {
    std::fprintf(stderr,
                 "virtual crossover prepare/plan mismatch n=%zu radius=%zu "
                 "active=%zu\n",
                 logical_count, radius, active);
    return false;
  }
  BackendEvidence cpu_evidence{};
  BackendEvidence metal_evidence{};
  cpu_evidence.prepare_us = microseconds(cpu_prepare_end - cpu_prepare_begin);
  metal_evidence.prepare_us =
      microseconds(metal_prepare_end - metal_prepare_begin);
  const bool cpu_first = ordinal % 2u == 0u;
  const bool cold_ok =
      cpu_first ? cold_run(*cpu, active, cpu_evidence.cold_us) &&
                      cold_run(*metal, active, metal_evidence.cold_us)
                : cold_run(*metal, active, metal_evidence.cold_us) &&
                      cold_run(*cpu, active, cpu_evidence.cold_us);
  if (!cold_ok) {
    std::fprintf(stderr,
                 "virtual crossover cold failed n=%zu radius=%zu active=%zu\n",
                 logical_count, radius, active);
    return false;
  }
  cpu->output_backing->reset(virtual_residency::TailPoison);
  metal->output_backing->reset(virtual_residency::TailPoison);
  const auto cpu_invalidated = cpu->output_backing->invalidate();
  const auto metal_invalidated = metal->output_backing->invalidate();
  const bool conditioned =
      cpu_invalidated && metal_invalidated &&
      run_abba(*cpu, *metal, active, nullptr, nullptr, false);
  const auto cpu_samples_started =
      conditioned ? cpu->pipeline.begin_samples()
                  : ::rund::compute::Status::fail(
                        ::rund::compute::Reason::PipelineInvalid);
  const auto metal_samples_started =
      conditioned ? metal->pipeline.begin_samples()
                  : ::rund::compute::Status::fail(
                        ::rund::compute::Reason::PipelineInvalid);
  if (!conditioned || !cpu_samples_started || !metal_samples_started) {
    if (cpu_samples_started) {
      static_cast<void>(cpu->pipeline.end_samples());
    }
    if (metal_samples_started) {
      static_cast<void>(metal->pipeline.end_samples());
    }
    std::fprintf(stderr,
                 "virtual crossover conditioning failed n=%zu radius=%zu "
                 "active=%zu\n",
                 logical_count, radius, active);
    return false;
  }
  const bool samples_ok = run_abba(*cpu, *metal, active, &cpu_evidence.samples,
                                   &metal_evidence.samples, true);
  const auto cpu_end = cpu->pipeline.end_samples();
  const auto metal_end = metal->pipeline.end_samples();
  if (!samples_ok || !cpu_end || !metal_end || !observe_output(*cpu, active) ||
      !observe_output(*metal, active)) {
    std::fprintf(stderr,
                 "virtual crossover sample/output failed n=%zu radius=%zu "
                 "active=%zu\n",
                 logical_count, radius, active);
    return false;
  }
  auto cpu_profile = cpu->pipeline.profile();
  auto metal_profile = metal->pipeline.profile();
  if (!cpu_profile || !metal_profile ||
      !valid_warm(*cpu, *cpu_profile, active) ||
      !valid_warm(*metal, *metal_profile, active)) {
    const auto print_terminal = [](const char *const label,
                                   const Profile *const profile) noexcept {
      if (profile == nullptr) {
        std::fprintf(stderr, " %s=missing", label);
        return;
      }
      const auto &stats = profile->execution();
      const auto &residency = stats.pipeline.residency;
      std::fprintf(
          stderr,
          " %s={q=%llu,receipt=%llu/%llu/%llu,submit=%llu,peak=%llu,"
          "transfer=%llu/"
          "%llu,backing=%llu,pagein=%llu,late=%llu,prefetch=%llu}",
          label, static_cast<unsigned long long>(residency.epoch_count),
          static_cast<unsigned long long>(residency.window_handoff_count),
          static_cast<unsigned long long>(residency.window_batch_count),
          static_cast<unsigned long long>(residency.window_queue_call_count),
          static_cast<unsigned long long>(stats.command_submits),
          static_cast<unsigned long long>(stats.command_inflight_peak),
          static_cast<unsigned long long>(stats.uploaded_bytes),
          static_cast<unsigned long long>(stats.downloaded_bytes),
          static_cast<unsigned long long>(residency.backing_read_bytes),
          static_cast<unsigned long long>(residency.page_in_bytes),
          static_cast<unsigned long long>(residency.late_page_count),
          static_cast<unsigned long long>(residency.prefetch_count));
    };
    std::fprintf(stderr,
                 "virtual crossover terminal evidence failed n=%zu radius=%zu "
                 "active=%zu",
                 logical_count, radius, active);
    print_terminal("cpu", cpu_profile ? &*cpu_profile : nullptr);
    print_terminal("metal", metal_profile ? &*metal_profile : nullptr);
    std::fputc('\n', stderr);
    return false;
  }
  cpu_evidence.output_hash = content_hash(cpu->observed, active);
  metal_evidence.output_hash = content_hash(metal->observed, active);
  if (cpu_evidence.output_hash != metal_evidence.output_hash ||
      cpu_profile->execution().output_hash != cpu_evidence.output_hash ||
      metal_profile->execution().output_hash != metal_evidence.output_hash ||
      cpu_profile->execution().graph_hash !=
          metal_profile->execution().graph_hash) {
    std::fprintf(stderr,
                 "virtual crossover identity mismatch n=%zu radius=%zu "
                 "active=%zu\n",
                 logical_count, radius, active);
    return false;
  }
  summarize(cpu_evidence);
  summarize(metal_evidence);
  cpu_evidence.profile.emplace(std::move(cpu_profile).value());
  metal_evidence.profile.emplace(std::move(metal_profile).value());
  print_point(logical_count, radius, active_ratio, active, *cpu, *metal,
              cpu_evidence, metal_evidence);
  return true;
}

} // namespace rund::measure::compute::virtual_crossover::detail

namespace rund::measure::compute {

bool MeasureVirtualCrossover() {
  std::size_t ordinal = 0u;
  bool ok = true;
  for (const std::size_t logical_count : virtual_crossover::LogicalCounts) {
    for (const std::size_t radius : virtual_crossover::Radii) {
      for (const virtual_crossover::ActiveRatio active_ratio :
           virtual_crossover::ActiveRatios) {
        ok = virtual_crossover::detail::measure_point(logical_count, radius,
                                                      active_ratio, ordinal) &&
             ok;
        ++ordinal;
      }
    }
  }
  return ok;
}

} // namespace rund::measure::compute
