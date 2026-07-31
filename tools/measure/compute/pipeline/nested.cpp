#include "model.hpp"

#include <numeric>

namespace rund::measure::compute {
namespace {

constexpr std::size_t Maximum = 516096u;
constexpr std::size_t Tile = 1024u;
constexpr std::size_t Inner = 64u;
constexpr std::size_t Outer = Maximum / Tile;
constexpr std::size_t Templates = Outer + Inner + 3u;
constexpr std::size_t Commands = Outer * (Inner + 2u);
constexpr std::size_t SerialSubmits = Outer * Inner;
constexpr std::size_t Domain = 64u;
constexpr std::uint32_t OuterSeed = 7u;
// One open/reset, one raw-status reset, one Seed status fold and one Seed
// preflight per outer window, the final Fold advance, canonicalization,
// terminal close, and final publication. Program dispatches stay authored.
constexpr std::uint64_t MetalNestedControlCommands =
    2u + 1u + Outer + Outer + 1u + 1u + 1u;

static_assert(Maximum % Tile == 0u);
static_assert(Outer == 504u);
static_assert(SerialSubmits == 32256u);
static_assert(Templates == 571u);
static_assert(Commands == 33264u);
static_assert(Inner % 2u == 0u);
static_assert(MetalNestedControlCommands == 1014u);

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
         stats.pipeline.rebinding_count == 0u;
}

[[nodiscard]] bool NestedEvidence(const Backend backend, const Stats &stats,
                                  const ::rund::compute::PipelinePlan &plan) {
  using ::rund::compute::PipelineNestedPhase;
  using ::rund::compute::PipelineStats;
  const bool physical_control =
      backend != Backend::Metal ||
      stats.pipeline.control_command_count == MetalNestedControlCommands;
  return physical_control && stats.backend == backend &&
         stats.command_submits == 1u &&
         stats.dispatches != 0u && stats.pipeline.step_count == 1u &&
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
  std::fputs(
      "window_repeat_columns,backend,command_path,status,maximum,"
      "outer_windows,tile,inner_iterations,samples,serial_first,"
      "nested_first,templates,commands,serial_wall_median_us,"
      "nested_wall_median_us,speedup,serial_command_submits,"
      "nested_command_submits,serial_dispatches,nested_dispatches,"
      "nested_control_commands,"
      "serial_warm_buffer_allocations,nested_warm_buffer_allocations,"
      "serial_warm_uploaded_bytes,nested_warm_uploaded_bytes,"
      "serial_warm_download_events,nested_warm_download_events,"
      "serial_warm_downloaded_bytes,nested_warm_downloaded_bytes,"
      "serial_warm_binding_mutation_count,"
      "nested_warm_binding_mutation_count,"
      "serial_fallback,"
      "nested_fallback,serial_result,nested_result,result_parity,warm_zero\n",
      stdout);
}

bool MeasureNestedRepeat(const Backend backend, const std::size_t samples) {
  using namespace ::rund::compute;
  if ((backend != Backend::Metal && backend != Backend::Vulkan) ||
      samples == 0u || samples % 2u != 0u) {
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
  if (!serial_seed || !serial_first || !serial_second || !serial_observed ||
      !queue || !domain || !count || !outer_seed || !nested_output) {
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
                                        write(*nested_output));
  const auto nested_plan = nested_builder.plan();
  auto nested = std::move(nested_builder).prepare();
  if (!nested_plan || !nested ||
      nested_plan->prepared_template_count != Templates ||
      nested_plan->prepared_command_count != Commands) {
    std::fprintf(stderr, "window repeat %s nested preparation failed\n",
                 Name(backend));
    return false;
  }

  std::vector<double> serial_wall;
  std::vector<double> nested_wall;
  serial_wall.reserve(samples);
  nested_wall.reserve(samples);
  WarmCounters serial_warm{};
  WarmCounters nested_warm{};
  std::uint64_t serial_warm_binding_mutation_count{};
  std::uint64_t nested_warm_binding_mutation_count{};
  ExecutionCounters serial_counters{};
  ExecutionCounters nested_counters{};
  Stats nested_stats{};

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

  const auto run_nested = [&](const bool timed) {
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
    if (!NestedEvidence(backend, nested_stats, *nested_plan)) {
      std::fprintf(stderr, "window repeat %s nested evidence failed\n",
                   Name(backend));
      return false;
    }
    nested_counters = {
        .command_submits = nested_stats.command_submits,
        .dispatches = nested_stats.dispatches,
    };
    ObserveWarm(nested_warm, nested_stats);
    ::rund::detail::counter::Accumulate(nested_warm_binding_mutation_count,
                                        nested_stats.pipeline.rebinding_count);
    if (timed) {
      nested_wall.push_back(
          std::chrono::duration<double, std::micro>(end - begin).count());
    }
    return true;
  };

  if (!run_serial(false) || !run_nested(false)) {
    return false;
  }
  std::size_t serial_first_count = 0u;
  std::size_t nested_first_count = 0u;
  for (std::size_t sample = 0u; sample < samples; ++sample) {
    const bool serial_then_nested = sample % 2u == 0u;
    const bool ok = serial_then_nested ? run_serial(true) && run_nested(true)
                                       : run_nested(true) && run_serial(true);
    if (!ok) {
      return false;
    }
    serial_first_count += serial_then_nested ? 1u : 0u;
    nested_first_count += serial_then_nested ? 0u : 1u;
  }

  if (!run_serial(false) || !run_nested(false)) {
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
  if (!nested->read(*nested_output, std::span<std::uint32_t>{nested_value})) {
    std::fprintf(stderr, "window repeat %s nested observation failed\n",
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
  const bool balanced =
      serial_first_count == samples / 2u && nested_first_count == samples / 2u;
  const bool parity = serial_result == expected &&
                      nested_value[0] == expected &&
                      serial_result == nested_value[0];
  constexpr bool serial_fallback = false;
  constexpr bool nested_fallback = false;
  const bool warm_zero = serial_warm.zero() && nested_warm.zero() &&
                         serial_warm_binding_mutation_count == 0u &&
                         nested_warm_binding_mutation_count == 0u;
  const bool contract =
      balanced && parity && warm_zero && !serial_fallback && !nested_fallback;
  const double serial_us = Median(serial_wall);
  const double nested_us = Median(nested_wall);
  const double speedup = nested_us == 0.0 ? 0.0 : serial_us / nested_us;

  std::printf(
      "window_repeat,%s,%s,%s,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,"
      "%.3f,%.3f,%.6f,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,"
      "%llu,%llu,%llu,%llu,%llu,%llu,%u,%u,%u,%u,%u,%u\n",
      Name(backend), CommandPath(backend), contract ? "ok" : "contract_failed",
      Maximum, Outer, Tile, Inner, samples, serial_first_count,
      nested_first_count, Templates, Commands, serial_us, nested_us, speedup,
      static_cast<unsigned long long>(serial_counters.command_submits),
      static_cast<unsigned long long>(nested_counters.command_submits),
      static_cast<unsigned long long>(serial_counters.dispatches),
      static_cast<unsigned long long>(nested_counters.dispatches),
      static_cast<unsigned long long>(
          nested_stats.pipeline.control_command_count),
      static_cast<unsigned long long>(serial_warm.buffer_allocations),
      static_cast<unsigned long long>(nested_warm.buffer_allocations),
      static_cast<unsigned long long>(serial_warm.uploaded_bytes),
      static_cast<unsigned long long>(nested_warm.uploaded_bytes),
      static_cast<unsigned long long>(serial_warm.download_events),
      static_cast<unsigned long long>(nested_warm.download_events),
      static_cast<unsigned long long>(serial_warm.downloaded_bytes),
      static_cast<unsigned long long>(nested_warm.downloaded_bytes),
      static_cast<unsigned long long>(serial_warm_binding_mutation_count),
      static_cast<unsigned long long>(nested_warm_binding_mutation_count),
      serial_fallback ? 1u : 0u, nested_fallback ? 1u : 0u, serial_result,
      nested_value[0], parity ? 1u : 0u, warm_zero ? 1u : 0u);
  return contract;
}

} // namespace rund::measure::compute
