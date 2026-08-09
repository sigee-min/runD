#include "../../allocation.hpp"
#include "../../pipeline/local.hpp"
#include "../local.hpp"
#include "local.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/pipeline/output.hpp"
#include "src/compute/pipeline/plan/contract.hpp"
#include "src/compute/pipeline/plan/publication.hpp"
#include "src/compute/pipeline/state.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <utility>

namespace rund::node::test_contract::window {
template <std::size_t WindowCount>
  requires(WindowCount >= 4u && WindowCount <= 6u)
[[nodiscard]] auto PublicationArityFoldProgram(Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(kOutputTile)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto outer, auto tile, auto count, auto ordinal,
                 auto witness) {
        (void)count;
        (void)ordinal;
        (void)witness;
        auto next = outer.combine(
            "window-publication-arity-fold", tile.reduce(Reduce::Sum),
            [](auto left, auto right) { return left + right; });
        auto first = tile.map("window-publication-arity-0",
                              [](auto value) { return value; });
        auto second = tile.map("window-publication-arity-1",
                               [](auto value) { return value + 1u; });
        auto third = tile.map("window-publication-arity-2",
                              [](auto value) { return value + 2u; });
        auto fourth = tile.map("window-publication-arity-3",
                               [](auto value) { return value + 3u; });
        if constexpr (WindowCount == 4u) {
          return outputs(next, first, second, third, fourth);
        } else {
          auto fifth = tile.map("window-publication-arity-4",
                                [](auto value) { return value + 4u; });
          if constexpr (WindowCount == 5u) {
            return outputs(next, first, second, third, fourth, fifth);
          } else {
            auto sixth = tile.map("window-publication-arity-5",
                                  [](auto value) { return value + 5u; });
            return outputs(next, first, second, third, fourth, fifth, sixth);
          }
        }
      })
      .compile();
}
template <std::size_t WindowCount, std::size_t Inner, class Seed, class Action,
          class Fold>
[[nodiscard]] int CheckPublicationArityCase(Device &device, const Seed &seed,
                                            const Action &action,
                                            const Fold &fold) {
  using namespace rund::compute;
  static_assert(WindowCount >= 4u && WindowCount <= 6u);
  static_assert(Inner <= 1u);
  constexpr std::array<std::uint32_t, 1u> initial{kOutputInitial};
  constexpr std::array<std::uint32_t, 1u> count_value{kOutputMaximum};
  constexpr std::array<std::uint32_t, 1u> witness_value{0u};

  auto outer = device.upload<std::uint32_t>(initial);
  auto values = device.upload<std::uint32_t>(kOutputValues);
  auto lanes = device.upload<std::uint32_t>(kOutputLanes);
  auto witness = device.upload<std::uint32_t>(witness_value);
  auto count = device.upload<std::uint32_t>(count_value);
  auto final = device.buffer<std::uint32_t>(1u);
  auto first = device.buffer<std::uint32_t>(kOutputMaximum);
  auto second = device.buffer<std::uint32_t>(kOutputMaximum);
  auto third = device.buffer<std::uint32_t>(kOutputMaximum);
  auto fourth = device.buffer<std::uint32_t>(kOutputMaximum);
  auto fifth = device.buffer<std::uint32_t>(kOutputMaximum);
  auto sixth = device.buffer<std::uint32_t>(kOutputMaximum);
  if (!outer || !values || !lanes || !witness || !count || !final || !first ||
      !second || !third || !fourth || !fifth || !sixth) {
    return 1;
  }

  const auto body = [&] {
    if constexpr (Inner == 0u) {
      return tile_repeat<0u>(seed, fold);
    } else {
      return tile_repeat<Inner>(seed, action, fold);
    }
  }();
  auto builder = pipeline(device);
  if constexpr (WindowCount == 4u) {
    builder.windows<kOutputMaximum, kOutputTile>(
        body, rund::compute::window(*count),
        read(*outer, *values, *lanes, *witness), write_final(*final),
        write_window(*first, *second, *third, *fourth));
  } else if constexpr (WindowCount == 5u) {
    builder.windows<kOutputMaximum, kOutputTile>(
        body, rund::compute::window(*count),
        read(*outer, *values, *lanes, *witness), write_final(*final),
        write_window(*first, *second, *third, *fourth, *fifth));
  } else {
    builder.windows<kOutputMaximum, kOutputTile>(
        body, rund::compute::window(*count),
        read(*outer, *values, *lanes, *witness), write_final(*final),
        write_window(*first, *second, *third, *fourth, *fifth, *sixth));
  }
  const auto plan = builder.plan();
  if (!plan || plan->outer_window_count != kOutputOuter ||
      plan->inner_iteration_count != Inner ||
      plan->publish_count != 1u + WindowCount * kOutputOuter) {
    if (!plan) {
      std::fprintf(stderr,
                   "window publication arity W=%zu N=%zu reason=%u "
                   "location=%u/%u/%u\n",
                   WindowCount, Inner, static_cast<unsigned>(plan.reason()),
                   plan.location().step, plan.location().iteration,
                   static_cast<unsigned>(plan.location().nested_phase));
    }
    return 2;
  }
  auto prepared = std::move(builder).prepare();
  if (!prepared) {
    std::fprintf(stderr,
                 "window publication arity prepare W=%zu N=%zu reason=%u "
                 "location=%u/%u/%u\n",
                 WindowCount, Inner, static_cast<unsigned>(prepared.reason()),
                 prepared.location().step, prepared.location().iteration,
                 static_cast<unsigned>(prepared.location().nested_phase));
    return 3;
  }
  const Status executed = prepared->run();
  if (!executed) {
    std::fprintf(stderr, "window publication arity run W=%zu N=%zu reason=%u\n",
                 WindowCount, Inner, static_cast<unsigned>(executed.reason()));
    return 3;
  }
  std::array<std::uint32_t, kOutputMaximum> observed{};
  const auto &last = [&]() -> const rund::compute::Buffer<std::uint32_t> & {
    if constexpr (WindowCount == 4u) {
      return *fourth;
    } else if constexpr (WindowCount == 5u) {
      return *fifth;
    } else {
      return *sixth;
    }
  }();
  if (!prepared->read(last, observed)) {
    return 4;
  }
  for (std::size_t index = 0u; index < observed.size(); ++index) {
    if (observed[index] !=
        kOutputValues[index] + static_cast<std::uint32_t>(WindowCount - 1u)) {
      return 5;
    }
  }
  return 0;
}

template <class Seed, class Action>
[[nodiscard]] int CheckPublicationArity(Device &device, const Seed &seed,
                                        const Action &action) {
  auto four = PublicationArityFoldProgram<4u>(device);
  auto five = PublicationArityFoldProgram<5u>(device);
  auto six = PublicationArityFoldProgram<6u>(device);
  if (!four || !five || !six) {
    return 1;
  }
  if (const int result =
          CheckPublicationArityCase<4u, 0u>(device, seed, action, *four);
      result != 0) {
    return 10 + result;
  }
  if (const int result =
          CheckPublicationArityCase<5u, 0u>(device, seed, action, *five);
      result != 0) {
    return 20 + result;
  }
  if (const int result =
          CheckPublicationArityCase<6u, 0u>(device, seed, action, *six);
      result != 0) {
    return 30 + result;
  }
  if (const int result =
          CheckPublicationArityCase<4u, 1u>(device, seed, action, *four);
      result != 0) {
    return 40 + result;
  }
  if (const int result =
          CheckPublicationArityCase<5u, 1u>(device, seed, action, *five);
      result != 0) {
    return 50 + result;
  }
  if (const int result =
          CheckPublicationArityCase<6u, 1u>(device, seed, action, *six);
      result != 0) {
    return 60 + result;
  }

  constexpr std::array<std::uint32_t, 1u> initial{kOutputInitial};
  constexpr std::array<std::uint32_t, 1u> count_value{kOutputMaximum};
  constexpr std::array<std::uint32_t, 1u> witness_value{0u};
  auto outer = device.upload<std::uint32_t>(initial);
  auto values = device.upload<std::uint32_t>(kOutputValues);
  auto lanes = device.upload<std::uint32_t>(kOutputLanes);
  auto witness = device.upload<std::uint32_t>(witness_value);
  auto count = device.upload<std::uint32_t>(count_value);
  auto final = device.buffer<std::uint32_t>(1u);
  auto first = device.buffer<std::uint32_t>(kOutputMaximum);
  auto second = device.buffer<std::uint32_t>(kOutputMaximum);
  auto third = device.buffer<std::uint32_t>(kOutputMaximum);
  auto fourth = device.buffer<std::uint32_t>(kOutputMaximum);
  auto fifth = device.buffer<std::uint32_t>(kOutputMaximum);
  if (!outer || !values || !lanes || !witness || !count || !final || !first ||
      !second || !third || !fourth || !fifth) {
    return 71;
  }
  using namespace rund::compute::detail;
  const std::array<ResourceView, 4u> inputs{
      BufferAccess::view(*outer, ResourceAccess::Read),
      BufferAccess::view(*values, ResourceAccess::Read),
      BufferAccess::view(*lanes, ResourceAccess::Read),
      BufferAccess::view(*witness, ResourceAccess::Read),
  };
  const std::array<ResourceView, 1u> finals{
      BufferAccess::view(*final, ResourceAccess::Write),
  };
  const std::array<ResourceView, 5u> windows{
      BufferAccess::view(*first, ResourceAccess::Write),
      BufferAccess::view(*second, ResourceAccess::Write),
      BufferAccess::view(*third, ResourceAccess::Write),
      BufferAccess::view(*fourth, ResourceAccess::Write),
      BufferAccess::view(*fifth, ResourceAccess::Write),
  };
  auto build = make_pipeline(DeviceAccess::state(device));
  append_pipeline_window_repeat(
      build, ProgramAccess::state(seed), {}, ProgramAccess::state(*five),
      BufferAccess::view(*count, ResourceAccess::Read), inputs, finals, windows,
      kOutputMaximum, kOutputTile, 0u, rund::compute::NoWindowTerminal, 1u);
  if (build == nullptr || build->failure != Reason::Ok ||
      build->publications.empty()) {
    return 72;
  }
  auto *terminal =
      std::get_if<PipelineBuildTerminalPublication>(&build->publications[0u]);
  if (terminal == nullptr) {
    return 73;
  }
  terminal->edge.output.value = 5u;
  const auto invalid = plan_pipeline(build);
  const rund::compute::Location location = invalid.location();
  if (invalid || invalid.reason() != Reason::PipelineInvalid ||
      location.step != 0u || location.iteration != 0u ||
      location.nested_phase != rund::compute::PipelineNestedPhase::Fold) {
    return 74;
  }
  return 0;
}

template <class Seed, class Fold>
[[nodiscard]] int CheckBuildAllocationRollback(Device &device, const Seed &seed,
                                               const Fold &fold) {
  using namespace rund::compute;
  using namespace rund::compute::detail;
  constexpr std::array<std::uint32_t, 1u> scalar{kOutputInitial};
  constexpr std::array<std::uint32_t, 1u> count_value{kOutputMaximum};
  constexpr std::array<std::uint32_t, 1u> witness_value{0u};
  auto outer = device.upload<std::uint32_t>(scalar);
  auto values = device.upload<std::uint32_t>(kOutputValues);
  auto lanes = device.upload<std::uint32_t>(kOutputLanes);
  auto witness = device.upload<std::uint32_t>(witness_value);
  auto count = device.upload<std::uint32_t>(count_value);
  auto final = device.buffer<std::uint32_t>(1u);
  auto window = device.buffer<std::uint32_t>(kOutputMaximum);
  if (!outer || !values || !lanes || !witness || !count || !final || !window) {
    return 1;
  }

  const std::array<ResourceView, 4u> inputs{
      BufferAccess::view(*outer, ResourceAccess::Read),
      BufferAccess::view(*values, ResourceAccess::Read),
      BufferAccess::view(*lanes, ResourceAccess::Read),
      BufferAccess::view(*witness, ResourceAccess::Read),
  };
  const std::array<ResourceView, 1u> finals{
      BufferAccess::view(*final, ResourceAccess::Write),
  };
  const std::array<ResourceView, 1u> windows{
      BufferAccess::view(*window, ResourceAccess::Write),
  };
  auto build = make_pipeline(DeviceAccess::state(device));
  if (build == nullptr) {
    return 2;
  }

  // Keep a non-empty prefix so the failure proves exact rollback rather than
  // merely clearing a newly-created build.
  build->steps.push_back(PipelineBuildStep{});
  build->internals.push_back(PipelineInternal{});
  build->publications.push_back(PipelineBuildTerminalPublication{});
  build->window_controls.push_back(PipelineBuildWindowControl{});
  build->nested_windows.push_back(PipelineBuildNestedWindow{});
  build->binding_count = 3u;

  // Leave exactly one allocation-free internal slot. The first nested
  // internal append must consume that slot and the second must grow the
  // vector, independent of the implementation's vector growth factor.
  build->internals.reserve(build->internals.size() + 1u);
  while (build->internals.size() + 1u < build->internals.capacity()) {
    build->internals.push_back(PipelineInternal{});
  }
  const std::size_t steps = build->steps.size();
  const std::size_t internals = build->internals.size();
  const std::size_t publications = build->publications.size();
  const std::size_t window_controls = build->window_controls.size();
  const std::size_t nested_windows = build->nested_windows.size();
  const std::size_t bindings = build->binding_count;

  // Seven non-empty local route vectors allocate first. The next allocation
  // is the second internal append, after the first append consumed the sole
  // spare slot, so this necessarily exercises partial-mutation rollback.
  node_compute_allocation::FailAfter(7u);
  append_pipeline_window_repeat(
      build, ProgramAccess::state(seed), {}, ProgramAccess::state(fold),
      BufferAccess::view(*count, ResourceAccess::Read), inputs, finals, windows,
      kOutputMaximum, kOutputTile, 0u, NoWindowTerminal, 1u);
  node_compute_allocation::ClearFailure();
  if (build->failure != Reason::PipelineCapacity ||
      build->steps.size() != steps || build->internals.size() != internals ||
      build->publications.size() != publications ||
      build->window_controls.size() != window_controls ||
      build->nested_windows.size() != nested_windows ||
      build->binding_count != bindings) {
    return 3;
  }

  build->failure = Reason::Ok;
  append_pipeline_window_repeat(
      build, ProgramAccess::state(seed), {}, ProgramAccess::state(fold),
      BufferAccess::view(*count, ResourceAccess::Read), inputs, finals, windows,
      kOutputMaximum, kOutputTile, 0u, NoWindowTerminal, 1u);
  return build->failure == Reason::Ok && build->steps.size() > steps &&
                 build->internals.size() > internals &&
                 build->publications.size() > publications &&
                 build->window_controls.size() == window_controls + 1u &&
                 build->nested_windows.size() == nested_windows + 1u &&
                 build->binding_count > bindings
             ? 0
             : 4;
}

[[nodiscard]] bool
OutputWarmSetupClean(const rund::compute::Stats &stats) noexcept {
  return stats.pipeline_compiles == 0u && stats.buffer_allocations == 0u &&
         stats.pipeline_cache_evictions == 0u &&
         stats.descriptor_pool_creations == 0u &&
         stats.descriptor_set_allocations == 0u && stats.uploaded_bytes == 0u &&
         stats.download_events == 0u && stats.downloaded_bytes == 0u;
}

[[nodiscard]] bool PublicationFingerprintV3Golden() {
  using namespace rund::compute::detail;
  const auto view =
      [](const std::uint32_t ordinal, const std::uint64_t backing_bytes,
         const std::uint64_t offset_bytes, const std::uint64_t count,
         const std::uint64_t stride_bytes, const std::uint32_t usage) {
        return PipelinePublicationViewPlan{
            .identity =
                PipelinePublicationViewIdentity{
                    .backing_bytes = backing_bytes,
                    .offset_bytes = offset_bytes,
                    .count = count,
                    .stride_bytes = stride_bytes,
                    .element_bytes = sizeof(std::uint32_t),
                    .resource_ordinal = ordinal,
                    .usage = usage,
                },
            .type = Type::U32,
        };
      };

  PipelineTerminalPublicationPlan terminal{
      .sources =
          {
              view(2u, 64u, 0u, 1u, 4u, rund::kernel::kResidentUsageRead),
              view(3u, 64u, 0u, 1u, 4u, rund::kernel::kResidentUsageRead),
              view(4u, 64u, 0u, 1u, 4u, rund::kernel::kResidentUsageRead),
          },
      .target =
          PipelinePublicationTargetPlan{
              .view =
                  view(7u, 64u, 8u, 1u, 8u, rund::kernel::kResidentUsageWrite),
          },
      .state = 1u,
      .output = {.value = 2u},
  };
  PipelineWindowControl terminal_control{.final = 2u};
  PipelineHash terminal_hash{};
  terminal_hash.number(1u);
  if (!mix_pipeline_publication_public_identity(terminal_hash, terminal,
                                                terminal_control)) {
    return false;
  }
  constexpr Fingerprint expected_terminal{
      .hi = 0xb5087f04bc866a06ull,
      .lo = 0x140961c5d5e8c583ull,
  };
  if (terminal_hash.finish() != expected_terminal) {
    return false;
  }

  // Public v3 serializes the selected canonical source in the source-ordinal
  // field. It deliberately does not add the private three-bank/final shape.
  PipelineTerminalPublicationPlan compatible = terminal;
  compatible.sources[0] =
      view(99u, 128u, 16u, 2u, 12u, rund::kernel::kResidentUsageRead);
  compatible.sources[1] = terminal.sources[2];
  PipelineWindowControl compatible_control{.final = 1u};
  PipelineHash compatible_hash{};
  compatible_hash.number(1u);
  if (!mix_pipeline_publication_public_identity(compatible_hash, compatible,
                                                compatible_control) ||
      compatible_hash.finish() != expected_terminal) {
    return false;
  }

  PipelineWindowPublicationPlan window{
      .source = view(9u, 16u, 0u, 4u, 4u, rund::kernel::kResidentUsageRead),
      .target =
          PipelinePublicationTargetPlan{
              .view = view(11u, 80u, 8u, 16u, 4u,
                           rund::kernel::kResidentUsageWrite),
          },
      .state = 2u,
      .output = {.value = 3u},
  };
  PipelineWindowControl window_control{
      .count = view(10u, 12u, 4u, 1u, 4u, rund::kernel::kResidentUsageRead),
      .maximum = 16u,
      .tile = 4u,
  };
  PipelineHash mixed_hash{};
  mixed_hash.number(2u);
  if (!mix_pipeline_publication_public_identity(mixed_hash, terminal,
                                                terminal_control) ||
      !mix_pipeline_publication_public_identity(mixed_hash, window,
                                                window_control)) {
    return false;
  }
  constexpr Fingerprint expected_mixed{
      .hi = 0xc9ed4d382b576784ull,
      .lo = 0x8c532fa716b8c443ull,
  };
  return mixed_hash.finish() == expected_mixed;
}

[[nodiscard]] bool PublicationSourceCoordinates() {
  using namespace rund::compute::detail;
  constexpr std::array<std::size_t, 4u> ordinary_steps{0u, 1u, 2u, 3u};
  constexpr std::array<std::size_t, 4u> nested_routes{0u, 1u, 2u, 1u};
  constexpr std::array<std::uint32_t, 4u> banks{
      PipelineWindow::first,
      PipelineWindow::second,
      PipelineWindow::first,
      PipelineWindow::second,
  };
  for (std::size_t index = 0u; index < ordinary_steps.size(); ++index) {
    const std::size_t bound = index + 1u;
    PipelineBuildState ordinary{};
    ordinary.steps.resize(bound);
    for (std::size_t iteration = 0u; iteration < bound; ++iteration) {
      ordinary.steps[iteration].iteration =
          static_cast<std::uint32_t>(iteration);
      ordinary.steps[iteration].iteration_bound =
          static_cast<std::uint32_t>(bound);
      ordinary.steps[iteration].window_control = {.value = 0u};
    }
    ordinary.window_controls.push_back(PipelineBuildWindowControl{
        .ordinary_step = {.value = 0u},
    });
    const PipelineBuildTerminalPublication publication{
        .edge =
            {
                .target = {},
                .control = {.value = 0u},
                .output = {},
            },
    };
    const auto final = resolve_build_window_final(ordinary, {.value = 0u});
    const auto source = resolve_publication_source(ordinary, publication);
    if (!final || !source ||
        final->source_step.value != ordinary_steps[index] ||
        source->step.value != ordinary_steps[index] ||
        final->bank != banks[index]) {
      return false;
    }

    PipelineBuildState nested{};
    const std::size_t fold_first = bound;
    nested.steps.resize(bound + 3u);
    for (std::size_t iteration = 0u; iteration < bound; ++iteration) {
      nested.steps[iteration].iteration = static_cast<std::uint32_t>(iteration);
      nested.steps[iteration].iteration_bound =
          static_cast<std::uint32_t>(bound);
      nested.steps[iteration].window_control = {.value = 0u};
      nested.steps[iteration].nested = 1u;
      nested.steps[iteration].route = PipelineRoute::NestedSeed;
    }
    for (std::size_t route = 0u; route < 3u; ++route) {
      nested.steps[fold_first + route].iteration =
          static_cast<std::uint32_t>(route);
      nested.steps[fold_first + route].iteration_bound = 3u;
      nested.steps[fold_first + route].window_control = {.value = 0u};
      nested.steps[fold_first + route].nested = 1u;
      nested.steps[fold_first + route].route = PipelineRoute::NestedFold;
    }
    nested.window_controls.push_back(PipelineBuildWindowControl{
        .nested = 1u,
    });
    node::accel::detail::NestedTemplateShape nested_shape{};
    if (!node::accel::detail::ProveNestedTemplateShape(
            0u, static_cast<std::uint32_t>(bound), 1u, 0u, nested_shape)) {
      return false;
    }
    nested.nested_windows.push_back(PipelineBuildNestedWindow{
        .shape = nested_shape,
    });
    const PipelineBuildTerminalPublication terminal{
        .edge =
            {
                .target = {},
                .control = {.value = 0u},
                .output = {},
            },
    };
    const PipelineBuildWindowPublication window{
        .edge =
            {
                .target = {},
                .control = {.value = 0u},
                .output = {},
            },
    };
    const auto nested_final = resolve_build_window_final(nested, {.value = 0u});
    const auto terminal_source = resolve_publication_source(nested, terminal);
    const auto window_source = resolve_publication_source(nested, window);
    if (!nested_final || !terminal_source || !window_source ||
        nested_final->source_step.value != fold_first + nested_routes[index] ||
        terminal_source->step.value != fold_first + nested_routes[index] ||
        window_source->step.value != fold_first ||
        nested_final->bank != banks[index]) {
      return false;
    }
  }
  return true;
}

int CheckOutputPublicationArity(Device &device, const OutputSeed &seed,
                                const OutputAction &action) {
  return CheckPublicationArity(device, seed, action);
}
int CheckOutputBuildAllocationRollback(Device &device, const OutputSeed &seed,
                                       const OutputFold &fold) {
  return CheckBuildAllocationRollback(device, seed, fold);
}

} // namespace rund::node::test_contract::window
