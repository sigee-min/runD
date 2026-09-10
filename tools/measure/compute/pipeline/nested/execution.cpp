#include "model.hpp"

namespace rund::measure::compute::nested_repeat {

bool RunSerial(Fixture &fixture, Measurements &measurements, const bool timed) {
  using namespace ::rund::compute;
  const auto reset =
      fixture.observe_program.run(fixture.serial_seed, fixture.serial_first);
  if (!reset) {
    std::fprintf(stderr, "window repeat %s serial reset failed\n",
                 Name(fixture.backend));
    return false;
  }

  ExecutionCounters counters{};
  const auto begin = Clock::now();
  for (std::size_t outer = 0u; outer < Outer; ++outer) {
    for (std::size_t inner = 0u; inner < Inner; ++inner) {
      Pipeline &step = fixture.serial_steps[outer * 2u + (inner & 1u)];
      const Status status = step.run();
      if (!status) {
        std::fprintf(stderr,
                     "window repeat %s serial execution failed "
                     "outer=%zu inner=%zu\n",
                     Name(fixture.backend), outer, inner);
        return false;
      }
      const Stats stats = step.stats();
      if (!SerialStepEvidence(fixture.backend, stats)) {
        std::fprintf(stderr,
                     "window repeat %s serial evidence failed "
                     "outer=%zu inner=%zu\n",
                     Name(fixture.backend), outer, inner);
        return false;
      }
      ::rund::detail::counter::Accumulate(counters.command_submits,
                                          stats.command_submits);
      ::rund::detail::counter::Accumulate(counters.dispatches,
                                          stats.dispatches);
      ObserveWarm(measurements.serial_warm, stats);
    }
  }
  const auto end = Clock::now();
  if (counters.command_submits != SerialSubmits ||
      counters.dispatches != SerialSubmits) {
    std::fprintf(stderr, "window repeat %s serial counters failed\n",
                 Name(fixture.backend));
    return false;
  }
  measurements.serial_counters = counters;
  if (timed) {
    measurements.serial_wall.push_back(
        std::chrono::duration<double, std::micro>(end - begin).count());
  }
  return true;
}

bool RunNested(Fixture &fixture, Measurements &measurements,
               std::vector<double> *const timings) {
  using namespace ::rund::compute;
  const auto begin = Clock::now();
  const Status status = fixture.nested.run();
  const auto end = Clock::now();
  if (!status) {
    std::fprintf(stderr, "window repeat %s nested execution failed: %.*s\n",
                 Name(fixture.backend), static_cast<int>(status.error().size()),
                 status.error().data());
    return false;
  }
  measurements.nested_stats = fixture.nested.stats();
  const Stats &stats = measurements.nested_stats;
  if (!NestedEvidence(fixture.backend, stats, fixture.nested_plan) ||
      stats.pipeline.sealed_repetition_count != 1u ||
      stats.pipeline.coalesced_repetition_count != 0u) {
    std::fprintf(
        stderr,
        "window repeat %s nested evidence failed: "
        "submits=%llu dispatches=%llu steps=%llu verified=%llu failed=%llu "
        "outer_failed=%llu inner_failed=%llu phase=%u outer=%llu/%llu "
        "inner=%llu/%llu templates=%llu commands=%llu controls=%llu "
        "plan_outer=%llu plan_tile=%llu plan_inner=%llu "
        "plan_templates=%llu plan_commands=%llu\n",
        Name(fixture.backend),
        static_cast<unsigned long long>(stats.command_submits),
        static_cast<unsigned long long>(stats.dispatches),
        static_cast<unsigned long long>(stats.pipeline.step_count),
        static_cast<unsigned long long>(stats.pipeline.verified_step_count),
        static_cast<unsigned long long>(stats.pipeline.failed_step_index),
        static_cast<unsigned long long>(stats.pipeline.failed_outer_window),
        static_cast<unsigned long long>(stats.pipeline.failed_inner_iteration),
        static_cast<unsigned>(stats.pipeline.failed_nested_phase),
        static_cast<unsigned long long>(
            stats.pipeline.executed_outer_window_count),
        static_cast<unsigned long long>(
            stats.pipeline.skipped_outer_window_count),
        static_cast<unsigned long long>(
            stats.pipeline.executed_inner_iteration_count),
        static_cast<unsigned long long>(
            stats.pipeline.skipped_inner_iteration_count),
        static_cast<unsigned long long>(stats.pipeline.prepared_template_count),
        static_cast<unsigned long long>(stats.pipeline.prepared_command_count),
        static_cast<unsigned long long>(stats.pipeline.control_command_count),
        static_cast<unsigned long long>(fixture.nested_plan.outer_window_count),
        static_cast<unsigned long long>(fixture.nested_plan.tile_capacity),
        static_cast<unsigned long long>(
            fixture.nested_plan.inner_iteration_count),
        static_cast<unsigned long long>(
            fixture.nested_plan.prepared_template_count),
        static_cast<unsigned long long>(
            fixture.nested_plan.prepared_command_count));
    return false;
  }
  measurements.nested_counters = {
      .command_submits = stats.command_submits,
      .dispatches = stats.dispatches,
  };
  ObserveWarm(measurements.nested_warm, stats);
  if (timings != nullptr) {
    timings->push_back(
        std::chrono::duration<double, std::micro>(end - begin).count());
  }
  return true;
}

bool RunRepeated(Fixture &fixture, Measurements &measurements,
                 std::vector<double> *const timings) {
  using namespace ::rund::compute;
  const bool timed = timings != nullptr;
  const auto inspect = [&](const std::size_t repetition) {
    measurements.repeated_stats = fixture.repeated.stats();
    const Stats &stats = measurements.repeated_stats;
    if (!NestedEvidence(fixture.backend, stats, fixture.repeated_plan) ||
        stats.pipeline.sealed_repetition_count != 1u ||
        stats.pipeline.coalesced_repetition_count != 0u) {
      std::fprintf(stderr,
                   "window repeat %s repeated evidence failed "
                   "repetition=%zu submits=%llu dispatches=%llu\n",
                   Name(fixture.backend), repetition,
                   static_cast<unsigned long long>(stats.command_submits),
                   static_cast<unsigned long long>(stats.dispatches));
      return false;
    }
    ObserveWarm(measurements.repeated_warm, stats);
    return true;
  };
  const std::uint64_t generation_before = fixture.repeated.generation();
  std::size_t succeeded = 0u;
  const auto begin = Clock::now();
  for (std::size_t repetition = 0u; repetition < Repetitions; ++repetition) {
    const Status status = fixture.repeated.run();
    if (!status) {
      std::fprintf(stderr,
                   "window repeat %s repeated execution failed "
                   "repetition=%zu: %.*s\n",
                   Name(fixture.backend), repetition,
                   static_cast<int>(status.error().size()),
                   status.error().data());
      return false;
    }
    ++succeeded;
    if (!timed && !inspect(repetition)) {
      return false;
    }
  }
  const auto end = Clock::now();
  if (timed && !inspect(Repetitions - 1u)) {
    return false;
  }
  if (succeeded != Repetitions ||
      fixture.repeated.generation() != generation_before + Repetitions) {
    std::fprintf(
        stderr,
        "window repeat %s repeated evidence failed "
        "succeeded=%zu generation=%llu/%llu submits=%llu dispatches=%llu\n",
        Name(fixture.backend), succeeded,
        static_cast<unsigned long long>(fixture.repeated.generation()),
        static_cast<unsigned long long>(generation_before + Repetitions),
        static_cast<unsigned long long>(
            measurements.repeated_stats.command_submits),
        static_cast<unsigned long long>(
            measurements.repeated_stats.dispatches));
    return false;
  }
  measurements.repeated_counters = {
      .command_submits =
          Repetitions * measurements.repeated_stats.command_submits,
      .dispatches = Repetitions * measurements.repeated_stats.dispatches,
  };
  if (timings != nullptr) {
    timings->push_back(
        std::chrono::duration<double, std::micro>(end - begin).count());
  }
  return true;
}

bool RunSealed(Fixture &fixture, Measurements &measurements,
               std::vector<double> *const timings) {
  using namespace ::rund::compute;
  const auto begin = Clock::now();
  const Status status = fixture.sealed.run();
  const auto end = Clock::now();
  if (!status) {
    std::fprintf(stderr, "window repeat %s sealed execution failed: %.*s\n",
                 Name(fixture.backend), static_cast<int>(status.error().size()),
                 status.error().data());
    return false;
  }
  measurements.sealed_stats = fixture.sealed.stats();
  const Stats &stats = measurements.sealed_stats;
  if (!NestedEvidence(fixture.backend, stats, fixture.sealed_plan) ||
      stats.pipeline.sealed_repetition_count != Repetitions ||
      stats.pipeline.coalesced_repetition_count != Repetitions - 1u) {
    std::fprintf(
        stderr,
        "window repeat %s sealed evidence failed: repetitions=%llu "
        "coalesced=%llu submits=%llu dispatches=%llu\n",
        Name(fixture.backend),
        static_cast<unsigned long long>(stats.pipeline.sealed_repetition_count),
        static_cast<unsigned long long>(
            stats.pipeline.coalesced_repetition_count),
        static_cast<unsigned long long>(stats.command_submits),
        static_cast<unsigned long long>(stats.dispatches));
    return false;
  }
  measurements.sealed_counters = {
      .command_submits = stats.command_submits,
      .dispatches = stats.dispatches,
  };
  ObserveWarm(measurements.sealed_warm, stats);
  if (timings != nullptr) {
    timings->push_back(
        std::chrono::duration<double, std::micro>(end - begin).count());
  }
  return true;
}

} // namespace rund::measure::compute::nested_repeat
