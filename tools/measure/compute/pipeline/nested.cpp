#include "model.hpp"

#include <numeric>

namespace rund::measure::compute {
namespace {

constexpr std::size_t Maximum = 516096u;
constexpr std::size_t Tile = 1024u;
constexpr std::size_t Inner = 64u;
constexpr std::size_t Repetitions = 256u;
constexpr std::size_t Outer = Maximum / Tile;
constexpr std::size_t Templates = Outer + Inner + 3u;
constexpr std::size_t Commands = Outer * (Inner + 2u);
constexpr std::size_t SerialSubmits = Outer * Inner;
constexpr std::size_t Domain = 64u;
constexpr std::uint32_t OuterSeed = 7u;
// The admitted aggregate owns one parallel tile-reduction command and one
// deterministic ordered finalize/publication command. A larger value proves
// that the canonical fallback stream ran instead of the aggregate hard cut.
constexpr std::uint64_t MetalNestedPhysicalCommands = 2u;

static_assert(Maximum % Tile == 0u);
static_assert(Outer == 504u);
static_assert(SerialSubmits == 32256u);
static_assert(Templates == 571u);
static_assert(Commands == 33264u);
static_assert(Inner % 2u == 0u);
static_assert(MetalNestedPhysicalCommands == 2u);
static_assert(Repetitions <= ::rund::compute::PipelineSealedRepetitionCapacity);

template <std::size_t Max, std::size_t Width>
[[nodiscard]] auto SeedProgram(::rund::compute::Device &device) {
  using namespace ::rund::compute;
  return on(device)
      .input<std::uint32_t>(Max)
      .zip_input<std::uint32_t>(Domain)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto queue, auto domain, auto total, auto ordinal) {
        auto current = resident<Max, Width>(total, ordinal);
        auto active_ordinals = queue.gather(current.items());
        return domain.gather(active_ordinals).reduce(Reduce::Sum);
      })
      .compile();
}

[[nodiscard]] auto ActionProgram(::rund::compute::Device &device) {
  using namespace ::rund::compute;
  return on(device)
      .map<std::uint32_t>("measure-window-repeat-action", 1u,
                          [](auto value) { return value + 1u; })
      .compile();
}

[[nodiscard]] auto FoldProgram(::rund::compute::Device &device) {
  using namespace ::rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto outer, auto tile) {
        return outer.combine(
            "measure-window-repeat-fold", tile,
            [](auto left, auto right) { return left + right; });
      })
      .compile();
}

// rebinding_count is the post-prepare binding-mutation diagnostic. The
// compute.window contract's frozen owner/View snapshot, not this zero alone,
// proves that prepared bindings remain identical across executions.
[[nodiscard]] bool SerialStepEvidence(const Backend backend,
                                      const Stats &stats) noexcept {
  using ::rund::compute::PipelineStats;
  return stats.backend == backend && stats.command_submits == 1u &&
         stats.dispatches == 1u && stats.pipeline.step_count == 1u &&
         stats.pipeline.verified_step_count == 1u &&
         stats.pipeline.failed_step_index == PipelineStats::no_failed_step &&
         stats.pipeline.sealed_repetition_count == 1u &&
         stats.pipeline.coalesced_repetition_count == 0u &&
         stats.pipeline.rebinding_count == 0u;
}

[[nodiscard]] bool NestedEvidence(const Backend backend, const Stats &stats,
                                  const ::rund::compute::PipelinePlan &plan) {
  using ::rund::compute::PipelineNestedPhase;
  using ::rund::compute::PipelineStats;
  const bool physical_shape =
      backend != Backend::Metal ||
      (stats.dispatches == MetalNestedPhysicalCommands &&
       stats.pipeline.control_command_count == 1u);
  return physical_shape && stats.backend == backend &&
         stats.command_submits == 1u && stats.dispatches != 0u &&
         stats.pipeline.step_count == 1u &&
         stats.pipeline.verified_step_count == 1u &&
         stats.pipeline.failed_step_index == PipelineStats::no_failed_step &&
         stats.pipeline.failed_outer_window == PipelineStats::no_coordinate &&
         stats.pipeline.failed_inner_iteration ==
             PipelineStats::no_coordinate &&
         stats.pipeline.failed_nested_phase == PipelineNestedPhase::None &&
         stats.pipeline.executed_outer_window_count == Outer &&
         stats.pipeline.skipped_outer_window_count == 0u &&
         stats.pipeline.executed_inner_iteration_count == SerialSubmits &&
         stats.pipeline.skipped_inner_iteration_count == 0u &&
         stats.pipeline.prepared_template_count == Templates &&
         stats.pipeline.prepared_command_count == Commands &&
         stats.pipeline.rebinding_count == 0u &&
         plan.outer_window_count == Outer && plan.tile_capacity == Tile &&
         plan.inner_iteration_count == Inner &&
         plan.prepared_template_count == Templates &&
         plan.prepared_command_count == Commands;
}

} // namespace

void PrintNestedRepeatColumns() {
  std::fputs("window_repeat_columns,backend,command_path,status,maximum,"
             "outer_windows,tile,inner_iterations,samples_per_pair,"
             "serial_pair_serial_first,serial_pair_nested_first,"
             "latency_pair_nested_first,latency_pair_sealed_first,"
             "throughput_pair_repeated_first,"
             "throughput_pair_sealed_first,templates,commands,"
             "sealed_repetitions,serial_wall_median_us,"
             "serial_pair_nested_wall_median_us,"
             "latency_pair_nested_wall_median_us,"
             "latency_pair_sealed_wall_median_us,"
             "repeated_nested_wall_median_us,"
             "throughput_pair_sealed_wall_median_us,"
             "sealed_equivalent_median_us,nested_speedup,"
             "sealed_measured_throughput_speedup,"
             "sealed_single_execution_ratio,serial_command_submits,"
             "nested_command_submits,repeated_command_submits,"
             "sealed_command_submits,serial_dispatches,nested_dispatches,"
             "repeated_dispatches,sealed_dispatches,"
             "nested_control_commands,"
             "serial_warm_buffer_allocations,nested_warm_buffer_allocations,"
             "repeated_warm_buffer_allocations,"
             "sealed_warm_buffer_allocations,"
             "serial_warm_uploaded_bytes,nested_warm_uploaded_bytes,"
             "repeated_warm_uploaded_bytes,sealed_warm_uploaded_bytes,"
             "serial_warm_download_events,nested_warm_download_events,"
             "repeated_warm_download_events,sealed_warm_download_events,"
             "serial_warm_downloaded_bytes,nested_warm_downloaded_bytes,"
             "repeated_warm_downloaded_bytes,sealed_warm_downloaded_bytes,"
             "serial_warm_binding_mutation_count,"
             "nested_warm_binding_mutation_count,"
             "repeated_warm_binding_mutation_count,"
             "sealed_warm_binding_mutation_count,serial_fallback,"
             "nested_fallback,repeated_fallback,sealed_fallback,serial_result,"
             "nested_result,repeated_result,sealed_result,result_parity,"
             "warm_zero\n",
             stdout);
}

bool MeasureNestedRepeat(const Backend backend, const std::size_t samples) {
  using namespace ::rund::compute;
  if ((backend != Backend::Metal && backend != Backend::Vulkan) ||
      samples == 0u || samples % 4u != 0u) {
    std::fprintf(stderr, "window repeat measurement configuration invalid\n");
    return false;
  }

  std::vector<std::uint32_t> queue_values(Maximum);
  for (std::size_t index = 0u; index < queue_values.size(); ++index) {
    queue_values[index] = static_cast<std::uint32_t>(index % Domain);
  }
  std::array<std::uint32_t, Domain> domain_values{};
  for (std::size_t index = 0u; index < domain_values.size(); ++index) {
    domain_values[index] = static_cast<std::uint32_t>(3u * index + 1u);
  }
  std::vector<std::uint32_t> tile_seeds(Outer);
  for (std::size_t outer = 0u; outer < Outer; ++outer) {
    std::uint32_t sum = 0u;
    for (std::size_t offset = 0u; offset < Tile; ++offset) {
      const std::size_t index = outer * Tile + offset;
      sum += domain_values[queue_values[index]];
    }
    tile_seeds[outer] = sum;
  }
  const std::uint64_t expected_wide =
      OuterSeed +
      std::accumulate(tile_seeds.begin(), tile_seeds.end(), std::uint64_t{}) +
      SerialSubmits;
  if (expected_wide > std::numeric_limits<std::uint32_t>::max()) {
    std::fprintf(stderr, "window repeat oracle overflow\n");
    return false;
  }
  const std::uint32_t expected = static_cast<std::uint32_t>(expected_wide);

  auto device = ::rund::compute::open(TargetFor(backend));
  if (!device) {
    std::fprintf(stderr, "window repeat %s open failed: %.*s\n", Name(backend),
                 static_cast<int>(device.error().size()),
                 device.error().data());
    return false;
  }
  auto seed_program = SeedProgram<Maximum, Tile>(*device);
  auto action_program = ActionProgram(*device);
  auto fold_program = FoldProgram(*device);
  auto observe_program =
      on(*device)
          .map<std::uint32_t>("measure-window-repeat-observe", Outer,
                              [](auto value) { return value; })
          .compile();
  if (!seed_program || !action_program || !fold_program || !observe_program) {
    std::fprintf(stderr, "window repeat %s program preparation failed\n",
                 Name(backend));
    return false;
  }

  auto serial_seed = device->upload<std::uint32_t>(tile_seeds);
  auto serial_first = device->buffer<std::uint32_t>(Outer);
  auto serial_second = device->buffer<std::uint32_t>(Outer);
  auto serial_observed = device->buffer<std::uint32_t>(Outer);
  const std::array<std::uint32_t, 1u> count_value{
      static_cast<std::uint32_t>(Maximum)};
  const std::array<std::uint32_t, 1u> outer_value{OuterSeed};
  const std::array<std::uint32_t, 1u> output_sentinel{0xA5A55A5Au};
  auto queue = device->upload<std::uint32_t>(queue_values);
  auto domain = device->upload<std::uint32_t>(domain_values);
  auto count = device->upload<std::uint32_t>(count_value);
  auto outer_seed = device->upload<std::uint32_t>(outer_value);
  auto nested_output = device->upload<std::uint32_t>(output_sentinel);
  auto repeated_output = device->upload<std::uint32_t>(output_sentinel);
  auto sealed_output = device->upload<std::uint32_t>(output_sentinel);
  if (!serial_seed || !serial_first || !serial_second || !serial_observed ||
      !queue || !domain || !count || !outer_seed || !nested_output ||
      !repeated_output || !sealed_output) {
    std::fprintf(stderr, "window repeat %s buffer preparation failed\n",
                 Name(backend));
    return false;
  }

  std::vector<Pipeline> serial_steps;
  serial_steps.reserve(Outer * 2u);
  for (std::size_t outer = 0u; outer < Outer; ++outer) {
    auto first = serial_first->view(outer, 1u);
    auto second = serial_second->view(outer, 1u);
    if (!first || !second) {
      std::fprintf(stderr, "window repeat %s serial view failed\n",
                   Name(backend));
      return false;
    }
    auto forward = pipeline(*device)
                       .then(*action_program, read(*first), write(*second))
                       .prepare();
    auto backward = pipeline(*device)
                        .then(*action_program, read(*second), write(*first))
                        .prepare();
    if (!forward || !backward) {
      std::fprintf(stderr,
                   "window repeat %s serial pipeline preparation failed "
                   "outer=%zu\n",
                   Name(backend), outer);
      return false;
    }
    serial_steps.emplace_back(std::move(*forward));
    serial_steps.emplace_back(std::move(*backward));
  }

  const auto body =
      tile_repeat<Inner>(*seed_program, *action_program, *fold_program);
  auto nested_builder = pipeline(*device);
  nested_builder.windows<Maximum, Tile>(body, window(*count),
                                        read(*outer_seed, *queue, *domain),
                                        write_final(*nested_output));
  const auto nested_plan = nested_builder.plan();
  auto nested = std::move(nested_builder).prepare();
  auto repeated_builder = pipeline(*device);
  repeated_builder.windows<Maximum, Tile>(body, window(*count),
                                          read(*outer_seed, *queue, *domain),
                                          write_final(*repeated_output));
  const auto repeated_plan = repeated_builder.plan();
  auto repeated = std::move(repeated_builder).prepare();
  auto sealed_builder = pipeline(*device);
  sealed_builder.sealed_repetitions<Repetitions>().windows<Maximum, Tile>(
      body, window(*count), read(*outer_seed, *queue, *domain),
      write_final(*sealed_output));
  const auto sealed_plan = sealed_builder.plan();
  auto sealed = std::move(sealed_builder).prepare();
  if (!nested_plan || !nested ||
      nested_plan->prepared_template_count != Templates ||
      nested_plan->prepared_command_count != Commands || !repeated_plan ||
      !repeated || *repeated_plan != *nested_plan || !sealed_plan || !sealed ||
      *sealed_plan != *nested_plan) {
    std::fprintf(stderr, "window repeat %s nested preparation failed\n",
                 Name(backend));
    return false;
  }

  std::vector<double> serial_wall;
  std::vector<double> serial_pair_nested_wall;
  std::vector<double> latency_pair_nested_wall;
  std::vector<double> latency_pair_sealed_wall;
  std::vector<double> repeated_wall;
  std::vector<double> throughput_pair_sealed_wall;
  serial_wall.reserve(samples);
  serial_pair_nested_wall.reserve(samples);
  latency_pair_nested_wall.reserve(samples);
  latency_pair_sealed_wall.reserve(samples);
  repeated_wall.reserve(samples);
  throughput_pair_sealed_wall.reserve(samples);
  WarmCounters serial_warm{};
  WarmCounters nested_warm{};
  WarmCounters repeated_warm{};
  WarmCounters sealed_warm{};
  std::uint64_t serial_warm_binding_mutation_count{};
  std::uint64_t nested_warm_binding_mutation_count{};
  std::uint64_t repeated_warm_binding_mutation_count{};
  std::uint64_t sealed_warm_binding_mutation_count{};
  ExecutionCounters serial_counters{};
  ExecutionCounters nested_counters{};
  ExecutionCounters repeated_counters{};
  ExecutionCounters sealed_counters{};
  Stats nested_stats{};
  Stats repeated_stats{};
  Stats sealed_stats{};

  const auto reset_serial = [&]() {
    const auto reset = observe_program->run(*serial_seed, *serial_first);
    if (!reset) {
      std::fprintf(stderr, "window repeat %s serial reset failed\n",
                   Name(backend));
      return false;
    }
    return true;
  };

  const auto run_serial = [&](const bool timed) {
    if (!reset_serial()) {
      return false;
    }
    ExecutionCounters counters{};
    const auto begin = Clock::now();
    for (std::size_t outer = 0u; outer < Outer; ++outer) {
      for (std::size_t inner = 0u; inner < Inner; ++inner) {
        Pipeline &step = serial_steps[outer * 2u + (inner & 1u)];
        const Status status = step.run();
        if (!status) {
          std::fprintf(stderr,
                       "window repeat %s serial execution failed "
                       "outer=%zu inner=%zu\n",
                       Name(backend), outer, inner);
          return false;
        }
        const Stats stats = step.stats();
        if (!SerialStepEvidence(backend, stats)) {
          std::fprintf(stderr,
                       "window repeat %s serial evidence failed "
                       "outer=%zu inner=%zu\n",
                       Name(backend), outer, inner);
          return false;
        }
        ::rund::detail::counter::Accumulate(counters.command_submits,
                                            stats.command_submits);
        ::rund::detail::counter::Accumulate(counters.dispatches,
                                            stats.dispatches);
        ObserveWarm(serial_warm, stats);
        ::rund::detail::counter::Accumulate(serial_warm_binding_mutation_count,
                                            stats.pipeline.rebinding_count);
      }
    }
    const auto end = Clock::now();
    if (counters.command_submits != SerialSubmits ||
        counters.dispatches != SerialSubmits) {
      std::fprintf(stderr, "window repeat %s serial counters failed\n",
                   Name(backend));
      return false;
    }
    serial_counters = counters;
    if (timed) {
      serial_wall.push_back(
          std::chrono::duration<double, std::micro>(end - begin).count());
    }
    return true;
  };

  const auto run_nested = [&](std::vector<double> *const timings) {
    const auto begin = Clock::now();
    const Status status = nested->run();
    const auto end = Clock::now();
    if (!status) {
      std::fprintf(stderr, "window repeat %s nested execution failed: %.*s\n",
                   Name(backend), static_cast<int>(status.error().size()),
                   status.error().data());
      return false;
    }
    nested_stats = nested->stats();
    if (!NestedEvidence(backend, nested_stats, *nested_plan) ||
        nested_stats.pipeline.sealed_repetition_count != 1u ||
        nested_stats.pipeline.coalesced_repetition_count != 0u) {
      std::fprintf(
          stderr,
          "window repeat %s nested evidence failed: "
          "submits=%llu dispatches=%llu steps=%llu verified=%llu failed=%llu "
          "outer_failed=%llu inner_failed=%llu phase=%u outer=%llu/%llu "
          "inner=%llu/%llu templates=%llu commands=%llu controls=%llu "
          "plan_outer=%llu plan_tile=%llu plan_inner=%llu "
          "plan_templates=%llu plan_commands=%llu rebind=%llu\n",
          Name(backend),
          static_cast<unsigned long long>(nested_stats.command_submits),
          static_cast<unsigned long long>(nested_stats.dispatches),
          static_cast<unsigned long long>(nested_stats.pipeline.step_count),
          static_cast<unsigned long long>(
              nested_stats.pipeline.verified_step_count),
          static_cast<unsigned long long>(
              nested_stats.pipeline.failed_step_index),
          static_cast<unsigned long long>(
              nested_stats.pipeline.failed_outer_window),
          static_cast<unsigned long long>(
              nested_stats.pipeline.failed_inner_iteration),
          static_cast<unsigned>(nested_stats.pipeline.failed_nested_phase),
          static_cast<unsigned long long>(
              nested_stats.pipeline.executed_outer_window_count),
          static_cast<unsigned long long>(
              nested_stats.pipeline.skipped_outer_window_count),
          static_cast<unsigned long long>(
              nested_stats.pipeline.executed_inner_iteration_count),
          static_cast<unsigned long long>(
              nested_stats.pipeline.skipped_inner_iteration_count),
          static_cast<unsigned long long>(
              nested_stats.pipeline.prepared_template_count),
          static_cast<unsigned long long>(
              nested_stats.pipeline.prepared_command_count),
          static_cast<unsigned long long>(
              nested_stats.pipeline.control_command_count),
          static_cast<unsigned long long>(nested_plan->outer_window_count),
          static_cast<unsigned long long>(nested_plan->tile_capacity),
          static_cast<unsigned long long>(nested_plan->inner_iteration_count),
          static_cast<unsigned long long>(nested_plan->prepared_template_count),
          static_cast<unsigned long long>(nested_plan->prepared_command_count),
          static_cast<unsigned long long>(
              nested_stats.pipeline.rebinding_count));
      return false;
    }
    nested_counters = {
        .command_submits = nested_stats.command_submits,
        .dispatches = nested_stats.dispatches,
    };
    ObserveWarm(nested_warm, nested_stats);
    ::rund::detail::counter::Accumulate(nested_warm_binding_mutation_count,
                                        nested_stats.pipeline.rebinding_count);
    if (timings != nullptr) {
      timings->push_back(
          std::chrono::duration<double, std::micro>(end - begin).count());
    }
    return true;
  };

  const auto run_repeated = [&](std::vector<double> *const timings) {
    const bool timed = timings != nullptr;
    const auto inspect = [&](const std::size_t repetition) {
      repeated_stats = repeated->stats();
      if (!NestedEvidence(backend, repeated_stats, *repeated_plan) ||
          repeated_stats.pipeline.sealed_repetition_count != 1u ||
          repeated_stats.pipeline.coalesced_repetition_count != 0u) {
        std::fprintf(
            stderr,
            "window repeat %s repeated evidence failed "
            "repetition=%zu submits=%llu dispatches=%llu\n",
            Name(backend), repetition,
            static_cast<unsigned long long>(repeated_stats.command_submits),
            static_cast<unsigned long long>(repeated_stats.dispatches));
        return false;
      }
      ObserveWarm(repeated_warm, repeated_stats);
      ::rund::detail::counter::Accumulate(
          repeated_warm_binding_mutation_count,
          repeated_stats.pipeline.rebinding_count);
      return true;
    };
    const std::uint64_t generation_before = repeated->generation();
    std::size_t succeeded = 0u;
    const auto begin = Clock::now();
    for (std::size_t repetition = 0u; repetition < Repetitions; ++repetition) {
      const Status status = repeated->run();
      if (!status) {
        std::fprintf(stderr,
                     "window repeat %s repeated execution failed "
                     "repetition=%zu: %.*s\n",
                     Name(backend), repetition,
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
        repeated->generation() != generation_before + Repetitions) {
      std::fprintf(
          stderr,
          "window repeat %s repeated evidence failed "
          "succeeded=%zu generation=%llu/%llu submits=%llu "
          "dispatches=%llu\n",
          Name(backend), succeeded,
          static_cast<unsigned long long>(repeated->generation()),
          static_cast<unsigned long long>(generation_before + Repetitions),
          static_cast<unsigned long long>(repeated_stats.command_submits),
          static_cast<unsigned long long>(repeated_stats.dispatches));
      return false;
    }
    repeated_counters = {
        .command_submits = Repetitions * repeated_stats.command_submits,
        .dispatches = Repetitions * repeated_stats.dispatches,
    };
    if (timings != nullptr) {
      timings->push_back(
          std::chrono::duration<double, std::micro>(end - begin).count());
    }
    return true;
  };

  const auto run_sealed = [&](std::vector<double> *const timings) {
    const auto begin = Clock::now();
    const Status status = sealed->run();
    const auto end = Clock::now();
    if (!status) {
      std::fprintf(stderr, "window repeat %s sealed execution failed: %.*s\n",
                   Name(backend), static_cast<int>(status.error().size()),
                   status.error().data());
      return false;
    }
    sealed_stats = sealed->stats();
    if (!NestedEvidence(backend, sealed_stats, *sealed_plan) ||
        sealed_stats.pipeline.sealed_repetition_count != Repetitions ||
        sealed_stats.pipeline.coalesced_repetition_count != Repetitions - 1u) {
      std::fprintf(
          stderr,
          "window repeat %s sealed evidence failed: repetitions=%llu "
          "coalesced=%llu submits=%llu dispatches=%llu\n",
          Name(backend),
          static_cast<unsigned long long>(
              sealed_stats.pipeline.sealed_repetition_count),
          static_cast<unsigned long long>(
              sealed_stats.pipeline.coalesced_repetition_count),
          static_cast<unsigned long long>(sealed_stats.command_submits),
          static_cast<unsigned long long>(sealed_stats.dispatches));
      return false;
    }
    sealed_counters = {
        .command_submits = sealed_stats.command_submits,
        .dispatches = sealed_stats.dispatches,
    };
    ObserveWarm(sealed_warm, sealed_stats);
    ::rund::detail::counter::Accumulate(sealed_warm_binding_mutation_count,
                                        sealed_stats.pipeline.rebinding_count);
    if (timings != nullptr) {
      timings->push_back(
          std::chrono::duration<double, std::micro>(end - begin).count());
    }
    return true;
  };

  if (!run_serial(false) || !run_nested(nullptr) || !run_repeated(nullptr) ||
      !run_sealed(nullptr)) {
    return false;
  }
  if (!run_serial(false) || !run_nested(nullptr) || !run_nested(nullptr) ||
      !run_serial(false)) {
    return false;
  }
  std::size_t serial_pair_serial_first = 0u;
  std::size_t serial_pair_nested_first = 0u;
  for (std::size_t sample = 0u; sample < samples; ++sample) {
    const bool serial_first = sample % 2u == 0u;
    serial_pair_serial_first += serial_first ? 1u : 0u;
    serial_pair_nested_first += serial_first ? 0u : 1u;
    const bool ok =
        serial_first ? run_serial(true) && run_nested(&serial_pair_nested_wall)
                     : run_nested(&serial_pair_nested_wall) && run_serial(true);
    if (!ok) {
      return false;
    }
  }

  if (!run_nested(nullptr) || !run_sealed(nullptr) || !run_sealed(nullptr) ||
      !run_nested(nullptr)) {
    return false;
  }
  std::size_t latency_pair_nested_first = 0u;
  std::size_t latency_pair_sealed_first = 0u;
  for (std::size_t sample = 0u; sample < samples; ++sample) {
    const bool nested_first = sample % 2u == 0u;
    latency_pair_nested_first += nested_first ? 1u : 0u;
    latency_pair_sealed_first += nested_first ? 0u : 1u;
    const bool ok = nested_first ? run_nested(&latency_pair_nested_wall) &&
                                       run_sealed(&latency_pair_sealed_wall)
                                 : run_sealed(&latency_pair_sealed_wall) &&
                                       run_nested(&latency_pair_nested_wall);
    if (!ok) {
      return false;
    }
  }

  if (!run_repeated(nullptr) || !run_sealed(nullptr) || !run_sealed(nullptr) ||
      !run_repeated(nullptr)) {
    return false;
  }
  std::size_t throughput_pair_repeated_first = 0u;
  std::size_t throughput_pair_sealed_first = 0u;
  for (std::size_t sample = 0u; sample < samples; ++sample) {
    const bool repeated_first = sample % 2u == 0u;
    throughput_pair_repeated_first += repeated_first ? 1u : 0u;
    throughput_pair_sealed_first += repeated_first ? 0u : 1u;
    const bool ok = repeated_first
                        ? run_repeated(&repeated_wall) &&
                              run_sealed(&throughput_pair_sealed_wall)
                        : run_sealed(&throughput_pair_sealed_wall) &&
                              run_repeated(&repeated_wall);
    if (!ok) {
      return false;
    }
  }

  if (!run_serial(false) || !run_nested(nullptr) || !run_repeated(nullptr) ||
      !run_sealed(nullptr)) {
    return false;
  }
  const auto observed = observe_program->run(*serial_first, *serial_observed);
  std::vector<std::uint32_t> serial_values(Outer);
  if (!observed || !observed->read(*serial_observed,
                                   std::span<std::uint32_t>{serial_values})) {
    std::fprintf(stderr, "window repeat %s serial observation failed\n",
                 Name(backend));
    return false;
  }
  std::array<std::uint32_t, 1u> nested_value{};
  std::array<std::uint32_t, 1u> repeated_value{};
  std::array<std::uint32_t, 1u> sealed_value{};
  if (!nested->read(*nested_output, std::span<std::uint32_t>{nested_value})) {
    std::fprintf(stderr, "window repeat %s nested observation failed\n",
                 Name(backend));
    return false;
  }
  if (!repeated->read(*repeated_output,
                      std::span<std::uint32_t>{repeated_value})) {
    std::fprintf(stderr, "window repeat %s repeated observation failed\n",
                 Name(backend));
    return false;
  }
  if (!sealed->read(*sealed_output, std::span<std::uint32_t>{sealed_value})) {
    std::fprintf(stderr, "window repeat %s sealed observation failed\n",
                 Name(backend));
    return false;
  }
  const std::uint64_t serial_wide =
      OuterSeed + std::accumulate(serial_values.begin(), serial_values.end(),
                                  std::uint64_t{});
  const std::uint32_t serial_result =
      serial_wide <= std::numeric_limits<std::uint32_t>::max()
          ? static_cast<std::uint32_t>(serial_wide)
          : 0u;
  const bool balanced = serial_pair_serial_first == samples / 2u &&
                        serial_pair_nested_first == samples / 2u &&
                        latency_pair_nested_first == samples / 2u &&
                        latency_pair_sealed_first == samples / 2u &&
                        throughput_pair_repeated_first == samples / 2u &&
                        throughput_pair_sealed_first == samples / 2u;
  const bool parity =
      serial_result == expected && nested_value[0] == expected &&
      repeated_value[0] == expected && sealed_value[0] == expected &&
      serial_result == nested_value[0] && serial_result == repeated_value[0] &&
      serial_result == sealed_value[0];
  constexpr bool serial_fallback = false;
  constexpr bool nested_fallback = false;
  constexpr bool repeated_fallback = false;
  constexpr bool sealed_fallback = false;
  const bool warm_zero = serial_warm.zero() && nested_warm.zero() &&
                         repeated_warm.zero() && sealed_warm.zero() &&
                         serial_warm_binding_mutation_count == 0u &&
                         nested_warm_binding_mutation_count == 0u &&
                         repeated_warm_binding_mutation_count == 0u &&
                         sealed_warm_binding_mutation_count == 0u;
  const bool contract =
      balanced && parity && warm_zero && !serial_fallback && !nested_fallback &&
      !repeated_fallback && !sealed_fallback &&
      repeated_counters.command_submits ==
          Repetitions * nested_counters.command_submits &&
      repeated_counters.dispatches ==
          Repetitions * nested_counters.dispatches &&
      sealed_counters.command_submits == nested_counters.command_submits &&
      sealed_counters.dispatches == nested_counters.dispatches;
  const double serial_us = Median(serial_wall);
  const double serial_pair_nested_us = Median(serial_pair_nested_wall);
  const double latency_pair_nested_us = Median(latency_pair_nested_wall);
  const double latency_pair_sealed_us = Median(latency_pair_sealed_wall);
  const double repeated_us = Median(repeated_wall);
  const double throughput_pair_sealed_us = Median(throughput_pair_sealed_wall);
  const double sealed_equivalent_us =
      throughput_pair_sealed_us / static_cast<double>(Repetitions);
  const double speedup =
      serial_pair_nested_us == 0.0 ? 0.0 : serial_us / serial_pair_nested_us;
  const double sealed_throughput_speedup =
      throughput_pair_sealed_us == 0.0
          ? 0.0
          : repeated_us / throughput_pair_sealed_us;
  const double sealed_single_execution_ratio =
      latency_pair_sealed_us == 0.0
          ? 0.0
          : latency_pair_nested_us / latency_pair_sealed_us;

  std::printf("window_repeat,%s,%s,%s,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,"
              "%zu,%zu,%zu,%zu",
              Name(backend), CommandPath(backend),
              contract ? "ok" : "contract_failed", Maximum, Outer, Tile, Inner,
              samples, serial_pair_serial_first, serial_pair_nested_first,
              latency_pair_nested_first, latency_pair_sealed_first,
              throughput_pair_repeated_first, throughput_pair_sealed_first,
              Templates, Commands, Repetitions);
  std::printf(",%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.6f,%.6f,%.6f,%.6f", serial_us,
              serial_pair_nested_us, latency_pair_nested_us,
              latency_pair_sealed_us, repeated_us, throughput_pair_sealed_us,
              sealed_equivalent_us, speedup, sealed_throughput_speedup,
              sealed_single_execution_ratio);
  std::printf(
      ",%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu",
      static_cast<unsigned long long>(serial_counters.command_submits),
      static_cast<unsigned long long>(nested_counters.command_submits),
      static_cast<unsigned long long>(repeated_counters.command_submits),
      static_cast<unsigned long long>(sealed_counters.command_submits),
      static_cast<unsigned long long>(serial_counters.dispatches),
      static_cast<unsigned long long>(nested_counters.dispatches),
      static_cast<unsigned long long>(repeated_counters.dispatches),
      static_cast<unsigned long long>(sealed_counters.dispatches),
      static_cast<unsigned long long>(
          nested_stats.pipeline.control_command_count));
  std::printf(",%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu",
              static_cast<unsigned long long>(serial_warm.buffer_allocations),
              static_cast<unsigned long long>(nested_warm.buffer_allocations),
              static_cast<unsigned long long>(repeated_warm.buffer_allocations),
              static_cast<unsigned long long>(sealed_warm.buffer_allocations),
              static_cast<unsigned long long>(serial_warm.uploaded_bytes),
              static_cast<unsigned long long>(nested_warm.uploaded_bytes),
              static_cast<unsigned long long>(repeated_warm.uploaded_bytes),
              static_cast<unsigned long long>(sealed_warm.uploaded_bytes));
  std::printf(",%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu",
              static_cast<unsigned long long>(serial_warm.download_events),
              static_cast<unsigned long long>(nested_warm.download_events),
              static_cast<unsigned long long>(repeated_warm.download_events),
              static_cast<unsigned long long>(sealed_warm.download_events),
              static_cast<unsigned long long>(serial_warm.downloaded_bytes),
              static_cast<unsigned long long>(nested_warm.downloaded_bytes),
              static_cast<unsigned long long>(repeated_warm.downloaded_bytes),
              static_cast<unsigned long long>(sealed_warm.downloaded_bytes));
  std::printf(
      ",%llu,%llu,%llu,%llu,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u\n",
      static_cast<unsigned long long>(serial_warm_binding_mutation_count),
      static_cast<unsigned long long>(nested_warm_binding_mutation_count),
      static_cast<unsigned long long>(repeated_warm_binding_mutation_count),
      static_cast<unsigned long long>(sealed_warm_binding_mutation_count),
      serial_fallback ? 1u : 0u, nested_fallback ? 1u : 0u,
      repeated_fallback ? 1u : 0u, sealed_fallback ? 1u : 0u, serial_result,
      nested_value[0], repeated_value[0], sealed_value[0], parity ? 1u : 0u,
      warm_zero ? 1u : 0u);
  return contract;
}

} // namespace rund::measure::compute
