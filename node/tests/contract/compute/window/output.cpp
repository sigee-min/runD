#include "../pipeline/local.hpp"
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
namespace {

constexpr std::size_t kMaximum = 16u;
constexpr std::size_t kTile = 4u;
constexpr std::size_t kOuter = kMaximum / kTile;
constexpr std::size_t kTemplates = kOuter + 3u;
constexpr std::size_t kCommands = 2u * kOuter;
constexpr std::size_t kWindowOffset = 2u;
constexpr std::size_t kWindowBacking = kMaximum + 4u;
constexpr std::uint32_t kInitial = 7u;
constexpr std::uint32_t kSentinel = 0xA5A55A5Au;
constexpr std::uint32_t kWindowBias = 1000u;
constexpr std::uint32_t kHighIndexValue = 0xF0000001u;
constexpr std::array<std::uint32_t, kMaximum> kValues{
    11u, 22u,  22u,  44u,  55u,  66u,  77u,  88u,
    99u, 110u, 121u, 132u, 143u, 154u, 165u, 0xF0000001u};
constexpr std::array<std::uint32_t, kTile> kLanes{0u, 1u, 2u, 3u};
constexpr std::array<std::uint32_t, 6u> kCounts{0u, 1u, 3u, 4u, 5u, 16u};

static_assert(kOuter == 4u);
static_assert(kTemplates == 7u);
static_assert(kCommands == 8u);

template <bool Fault> [[nodiscard]] auto SeedProgram(Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(kMaximum)
      .zip_input<std::uint32_t>(kTile)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto values, auto lanes, auto witness, auto total,
                 auto ordinal) {
        auto current = resident<kMaximum, kTile>(total, ordinal);
        auto indices =
            lanes.combine("window-output-index", current.base(),
                          [](auto lane, auto base) { return lane + base; });
        auto tile = values.gather(indices);
        auto produced = [&] {
          if constexpr (Fault) {
            auto fault =
                ordinal.map("window-output-seed-fault-index", [](auto value) {
                  return select(value == 1u, 1u, 0u);
                });
            auto checked = witness.gather(fault).scalar();
            return tile.combine(
                "window-output-seed-fault", checked,
                [](auto value, auto check) { return value + check * 0u; });
          } else {
            return tile.map("window-output-seed-tile",
                            [](auto value) { return value; });
          }
        }();
        auto count = current.count().map("window-output-seed-count",
                                         [](auto value) { return value; });
        auto canonical_ordinal = current.ordinal().map(
            "window-output-seed-ordinal", [](auto value) { return value; });
        auto retained = witness.map("window-output-seed-witness",
                                    [](auto value) { return value; });
        return outputs(produced, count, canonical_ordinal, retained);
      })
      .compile();
}

template <bool Fault> [[nodiscard]] auto ActionProgram(Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(kTile)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto tile, auto count, auto ordinal, auto witness) {
        auto produced = [&] {
          if constexpr (Fault) {
            auto fault =
                ordinal.map("window-output-action-fault-index", [](auto value) {
                  return select(value == 1u, 1u, 0u);
                });
            auto checked = witness.gather(fault).scalar();
            return tile.combine(
                "window-output-action-fault", checked,
                [](auto value, auto check) { return value + check * 0u; });
          } else {
            return tile.map("window-output-action-tile",
                            [](auto value) { return value; });
          }
        }();
        auto next_count = count.map("window-output-action-count",
                                    [](auto value) { return value; });
        auto next_ordinal = ordinal.map("window-output-action-ordinal",
                                        [](auto value) { return value; });
        auto next_witness = witness.map("window-output-action-witness",
                                        [](auto value) { return value; });
        return outputs(produced, next_count, next_ordinal, next_witness);
      })
      .compile();
}

template <bool TwoWindows = false, bool HighIndexOnly = false>
[[nodiscard]] auto FoldProgram(Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(kTile)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto outer, auto tile, auto count, auto ordinal,
                 auto witness) {
        (void)witness;
        auto active =
            tile.indices().combine("window-output-active", count.scalar(),
                                   [](auto lane, auto logical) {
                                     return select(lane < logical, 1u, 0u);
                                   });
        auto masked = tile.combine("window-output-mask", active,
                                   [](auto value, auto enabled) {
                                     return select(enabled != 0u, value, 0u);
                                   });
        auto sum = masked.reduce(Reduce::Sum);
        auto next =
            outer.combine("window-output-fold", sum,
                          [](auto state, auto value) { return state + value; });
        auto first = [&] {
          if constexpr (HighIndexOnly) {
            return tile.indices().combine(
                "window-output-high-index", ordinal.scalar(),
                [](auto lane, auto current) {
                  return select(lane == kTile - 1u,
                                select(current == kOuter - 1u, kHighIndexValue,
                                       kSentinel),
                                kSentinel);
                });
          } else {
            return tile.map("window-output-first",
                            [](auto value) { return value + kWindowBias; });
          }
        }();
        if constexpr (TwoWindows) {
          auto second = tile.combine(
              "window-output-second", active,
              [](auto value, auto enabled) { return value + enabled; });
          return outputs(next, first, second);
        } else {
          return outputs(next, first);
        }
      })
      .compile();
}

template <std::size_t WindowCount>
  requires(WindowCount >= 4u && WindowCount <= 6u)
[[nodiscard]] auto PublicationArityFoldProgram(Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(kTile)
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

[[nodiscard]] auto PriorityFoldProgram(Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(kTile)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto outer, auto tile, auto count, auto ordinal,
                 auto witness) {
        (void)count;
        (void)witness;
        auto targets = tile.indices().combine(
            "window-output-priority-target", ordinal.scalar(),
            [](auto lane, auto current) {
              // Outer 1 has two competing Scatter failures: target 4 is out
              // of range at source ordinal 1, while target 0 is duplicated at
              // source ordinal 2. The canonical failure key must retain the
              // higher-priority ordinal-1 range evidence even if the duplicate
              // is recorded first by a physically concurrent GPU lane.
              const auto failed =
                  select(lane == 1u, static_cast<std::uint32_t>(kTile),
                         select(lane == 2u, 0u, select(lane == 3u, 2u, lane)));
              return select(current == 1u, failed, lane);
            });
        auto sum = tile.reduce(Reduce::Sum);
        auto next =
            outer.combine("window-output-priority-fold", sum,
                          [](auto state, auto value) { return state + value; });
        auto scattered = tile.scatter(targets, {.count = kTile});
        return outputs(next, scattered);
      })
      .compile();
}

[[nodiscard]] auto ScatterSeedProgram(Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(kTile)
      .zip_input<std::uint32_t>(kTile)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto values, auto targets, auto total, auto ordinal) {
        (void)total;
        auto selected =
            values.combine("window-output-scatter-select", ordinal.scalar(),
                           [](auto value, auto outer) {
                             return select(outer == kOuter - 1u, value, 0u);
                           });
        auto retained = targets.map("window-output-scatter-target",
                                    [](auto target) { return target; });
        return outputs(selected, retained);
      })
      .compile();
}

[[nodiscard]] auto ScatterFoldProgram(Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(kTile)
      .zip_input<std::uint32_t>(kTile)
      .branch([](auto outer, auto values, auto targets) {
        auto sum = values.reduce(Reduce::Sum);
        auto next =
            outer.combine("window-output-scatter-fold", sum,
                          [](auto state, auto value) { return state + value; });
        auto scattered = values.scatter_reduce(targets, kTile, Reduce::Sum);
        return outputs(next, scattered);
      })
      .compile();
}

[[nodiscard]] auto DownstreamProgram(Device &device) {
  using namespace rund::compute;
  return on(device)
      .map<std::uint32_t>("window-output-downstream", kMaximum,
                          [](auto value) { return value; })
      .compile();
}

[[nodiscard]] auto CountAdvanceProgram(Device &device) {
  using namespace rund::compute;
  return on(device)
      .map<std::uint32_t>("window-output-count-advance", 1u,
                          [](auto value) { return value + kTile; })
      .compile();
}

[[nodiscard]] auto ZeroPrefixSeedProgram(Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(kTile)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto values, auto total, auto ordinal) {
        (void)ordinal;
        auto logical = total.map("window-output-zero-seed-count",
                                 [](auto value) { return value; });
        auto tile = values.map("window-output-zero-seed-tile",
                               [](auto value) { return value; });
        return outputs(logical, tile);
      })
      .compile();
}

[[nodiscard]] auto ZeroPrefixFoldProgram(Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(kTile)
      .branch([](auto outer, auto logical, auto tile) {
        (void)logical;
        auto next = outer.map("window-output-zero-fold-state",
                              [](auto value) { return value; });
        auto appended = tile.map("window-output-zero-fold-tile",
                                 [](auto value) { return value; });
        return outputs(next, appended);
      })
      .compile();
}

[[nodiscard]] constexpr std::uint32_t
ExpectedFinal(const std::array<std::uint32_t, kMaximum> &values,
              const std::uint32_t count) noexcept {
  std::uint32_t result = kInitial;
  for (std::size_t index = 0u; index < count; ++index) {
    result += values[index];
  }
  return result;
}

[[nodiscard]] constexpr std::array<std::uint32_t, kMaximum>
ExpectedWindow(const std::array<std::uint32_t, kMaximum> &values,
               const std::uint32_t count) noexcept {
  std::array<std::uint32_t, kMaximum> result{};
  result.fill(kSentinel);
  for (std::size_t index = 0u; index < count; ++index) {
    result[index] = values[index] + kWindowBias;
  }
  return result;
}

[[nodiscard]] constexpr std::array<std::uint32_t, kMaximum>
ExpectedHighIndexWindow() noexcept {
  std::array<std::uint32_t, kMaximum> result{};
  result.fill(kSentinel);
  result[kMaximum - 1u] = kHighIndexValue;
  return result;
}

template <std::size_t WindowCount, std::size_t Inner, class Seed, class Action,
          class Fold>
[[nodiscard]] int CheckPublicationArityCase(Device &device, const Seed &seed,
                                            const Action &action,
                                            const Fold &fold) {
  using namespace rund::compute;
  static_assert(WindowCount >= 4u && WindowCount <= 6u);
  static_assert(Inner <= 1u);
  constexpr std::array<std::uint32_t, 1u> initial{kInitial};
  constexpr std::array<std::uint32_t, 1u> count_value{kMaximum};
  constexpr std::array<std::uint32_t, 1u> witness_value{0u};

  auto outer = device.upload<std::uint32_t>(initial);
  auto values = device.upload<std::uint32_t>(kValues);
  auto lanes = device.upload<std::uint32_t>(kLanes);
  auto witness = device.upload<std::uint32_t>(witness_value);
  auto count = device.upload<std::uint32_t>(count_value);
  auto final = device.buffer<std::uint32_t>(1u);
  auto first = device.buffer<std::uint32_t>(kMaximum);
  auto second = device.buffer<std::uint32_t>(kMaximum);
  auto third = device.buffer<std::uint32_t>(kMaximum);
  auto fourth = device.buffer<std::uint32_t>(kMaximum);
  auto fifth = device.buffer<std::uint32_t>(kMaximum);
  auto sixth = device.buffer<std::uint32_t>(kMaximum);
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
    builder.windows<kMaximum, kTile>(
        body, rund::compute::window(*count),
        read(*outer, *values, *lanes, *witness), write_final(*final),
        write_window(*first, *second, *third, *fourth));
  } else if constexpr (WindowCount == 5u) {
    builder.windows<kMaximum, kTile>(
        body, rund::compute::window(*count),
        read(*outer, *values, *lanes, *witness), write_final(*final),
        write_window(*first, *second, *third, *fourth, *fifth));
  } else {
    builder.windows<kMaximum, kTile>(
        body, rund::compute::window(*count),
        read(*outer, *values, *lanes, *witness), write_final(*final),
        write_window(*first, *second, *third, *fourth, *fifth, *sixth));
  }
  const auto plan = builder.plan();
  if (!plan || plan->outer_window_count != kOuter ||
      plan->inner_iteration_count != Inner ||
      plan->publish_count != 1u + WindowCount * kOuter) {
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
  std::array<std::uint32_t, kMaximum> observed{};
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
        kValues[index] + static_cast<std::uint32_t>(WindowCount - 1u)) {
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

  constexpr std::array<std::uint32_t, 1u> initial{kInitial};
  constexpr std::array<std::uint32_t, 1u> count_value{kMaximum};
  constexpr std::array<std::uint32_t, 1u> witness_value{0u};
  auto outer = device.upload<std::uint32_t>(initial);
  auto values = device.upload<std::uint32_t>(kValues);
  auto lanes = device.upload<std::uint32_t>(kLanes);
  auto witness = device.upload<std::uint32_t>(witness_value);
  auto count = device.upload<std::uint32_t>(count_value);
  auto final = device.buffer<std::uint32_t>(1u);
  auto first = device.buffer<std::uint32_t>(kMaximum);
  auto second = device.buffer<std::uint32_t>(kMaximum);
  auto third = device.buffer<std::uint32_t>(kMaximum);
  auto fourth = device.buffer<std::uint32_t>(kMaximum);
  auto fifth = device.buffer<std::uint32_t>(kMaximum);
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
      kMaximum, kTile, 0u, rund::compute::NoWindowTerminal, 1u);
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

[[nodiscard]] bool WarmSetupClean(const rund::compute::Stats &stats) noexcept {
  return stats.pipeline_compiles == 0u && stats.buffer_allocations == 0u &&
         stats.pipeline_cache_evictions == 0u &&
         stats.descriptor_pool_creations == 0u &&
         stats.descriptor_set_allocations == 0u && stats.uploaded_bytes == 0u &&
         stats.download_events == 0u && stats.downloaded_bytes == 0u &&
         stats.pipeline.rebinding_count == 0u;
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

  // Public v3 serializes the selected canonical source in the legacy source
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

[[nodiscard]] int CheckOrdinaryAliasAuthority(Device &device) {
  using namespace rund::compute;
  using namespace rund::compute::detail;
  constexpr std::array<std::uint32_t, 1u> first_seed{10u};
  constexpr std::array<std::uint32_t, 1u> second_seed{20u};
  constexpr std::array<std::uint32_t, 1u> count_value{4u};
  auto program = on(device)
                     .input<std::uint32_t>(1u)
                     .zip_input<std::uint32_t>(1u)
                     .zip_input<std::uint32_t>(1u)
                     .zip_input<std::uint32_t>(1u)
                     .zip_input<std::uint32_t>(1u)
                     .branch([](auto first, auto second, auto alias, auto count,
                                auto ordinal) {
                       (void)alias;
                       (void)count;
                       (void)ordinal;
                       auto next_first =
                           first.map("window-ordinary-alias-first",
                                     [](auto value) { return value + 1u; });
                       auto next_second =
                           second.map("window-ordinary-alias-second",
                                      [](auto value) { return value + 2u; });
                       return outputs(next_first, next_second, next_first);
                     })
                     .compile();
  auto first = device.upload<std::uint32_t>(first_seed);
  auto second = device.upload<std::uint32_t>(second_seed);
  auto count = device.upload<std::uint32_t>(count_value);
  auto first_target = device.buffer<std::uint32_t>(1u);
  auto second_target = device.buffer<std::uint32_t>(1u);
  auto bad_target = device.buffer<std::uint32_t>(1u);
  if (!program || !first || !second || !count || !first_target ||
      !second_target || !bad_target) {
    return 1;
  }
  const std::array<ResourceView, 3u> inputs{
      BufferAccess::view(*first, ResourceAccess::Read),
      BufferAccess::view(*second, ResourceAccess::Read),
      BufferAccess::view(*first, ResourceAccess::Read),
  };
  const std::array<ResourceView, 3u> outputs{
      BufferAccess::view(*first_target, ResourceAccess::Write),
      BufferAccess::view(*second_target, ResourceAccess::Write),
      BufferAccess::view(*first_target, ResourceAccess::Write),
  };
  auto build = make_pipeline(DeviceAccess::state(device));
  append_pipeline_windows(build, ProgramAccess::state(*program),
                          BufferAccess::view(*count, ResourceAccess::Read),
                          inputs, outputs, 4u, 1u, NoWindowTerminal, 1u);
  if (build == nullptr || build->failure != Reason::Ok ||
      build->steps.size() != 4u || build->internals.size() != 5u ||
      build->publications.size() != 2u ||
      std::any_of(build->steps.begin(), build->steps.end(),
                  [](const auto &step) {
                    return step.outputs.size() != 3u ||
                           step.outputs[0u].owner != step.outputs[2u].owner ||
                           step.outputs[0u].owner == step.outputs[1u].owner;
                  })) {
    return 2;
  }
  for (const PipelineBuildPublication &authored : build->publications) {
    const auto *terminal =
        std::get_if<PipelineBuildTerminalPublication>(&authored);
    if (terminal == nullptr) {
      return 3;
    }
    const auto source = resolve_publication_source(*build, *terminal);
    const auto output =
        source ? resolve_build_output(*build, *source)
               : Result<PipelineBuildOutputProjection>::fail(source.reason());
    if (!source || !output || source->step.value != 3u ||
        output->source.value != terminal->edge.output.value) {
      return 3;
    }
  }
  const auto plan = plan_pipeline(build);
  auto prepared = prepare_pipeline(build);
  if (!plan || !prepared) {
    return 4;
  }
  constexpr std::array<std::uint32_t, 4u> counts{0u, 1u, 3u, 4u};
  for (const std::uint32_t active : counts) {
    const std::array<std::uint32_t, 1u> next_count{active};
    if (!rund_node_test_pipeline::Overwrite(*count, next_count) ||
        !run_pipeline(*prepared)) {
      return 5;
    }
    std::array<std::uint32_t, 1u> first_actual{};
    std::array<std::uint32_t, 1u> second_actual{};
    if (!read_pipeline_raw(*prepared, BufferAccess::state(*first_target),
                           Type::U32, FixedFormat{}, first_actual.data(),
                           sizeof(first_actual), first_actual.size()) ||
        !read_pipeline_raw(*prepared, BufferAccess::state(*second_target),
                           Type::U32, FixedFormat{}, second_actual.data(),
                           sizeof(second_actual), second_actual.size()) ||
        first_actual[0u] != first_seed[0u] + active ||
        second_actual[0u] != second_seed[0u] + 2u * active) {
      return 5;
    }
  }

  const std::array<ResourceView, 3u> mismatched{
      outputs[0u],
      outputs[1u],
      BufferAccess::view(*bad_target, ResourceAccess::Write),
  };
  auto rejected = make_pipeline(DeviceAccess::state(device));
  append_pipeline_windows(rejected, ProgramAccess::state(*program),
                          BufferAccess::view(*count, ResourceAccess::Read),
                          inputs, mismatched, 4u, 1u, NoWindowTerminal, 1u);
  const auto rejected_plan = plan_pipeline(rejected);
  if (rejected_plan ||
      rejected_plan.reason() != Reason::BindingAliasUnsupported) {
    return 6;
  }
  ResourceView invalid_count =
      BufferAccess::view(*count, ResourceAccess::Write);
  auto invalid_resident = make_pipeline(DeviceAccess::state(device));
  append_pipeline_windows(invalid_resident, ProgramAccess::state(*program),
                          invalid_count, inputs, mismatched, 4u, 1u,
                          NoWindowTerminal, 1u);
  const auto invalid_resident_plan = plan_pipeline(invalid_resident);
  if (invalid_resident_plan ||
      invalid_resident_plan.reason() != Reason::BindingInvalid) {
    return 7;
  }
  auto invalid_access_outputs = outputs;
  invalid_access_outputs[2u].access = ResourceAccess::Read;
  auto invalid_access = make_pipeline(DeviceAccess::state(device));
  append_pipeline_windows(invalid_access, ProgramAccess::state(*program),
                          BufferAccess::view(*count, ResourceAccess::Read),
                          inputs, invalid_access_outputs, 4u, 1u,
                          NoWindowTerminal, 1u);
  const auto invalid_access_plan = plan_pipeline(invalid_access);
  return !invalid_access_plan &&
                 invalid_access_plan.reason() == Reason::BindingInvalid
             ? 0
             : 8;
}

[[nodiscard]] int CheckAliasSubviewRouting(Device &device) {
  using namespace rund::compute;
  using namespace rund::compute::detail;
  constexpr std::size_t width = 4u;
  constexpr std::array<std::uint32_t, width> first_seed{1u, 2u, 3u, 4u};
  constexpr std::array<std::uint32_t, width> second_seed{10u, 20u, 30u, 40u};
  constexpr std::array<std::uint32_t, 1u> count_value{3u};
  constexpr std::array<std::uint32_t, 9u> backing_seed{
      kSentinel, kSentinel, kSentinel, kSentinel, kSentinel,
      kSentinel, kSentinel, kSentinel, kSentinel,
  };
  auto program = on(device)
                     .input<std::uint32_t>(width)
                     .zip_input<std::uint32_t>(width)
                     .zip_input<std::uint32_t>(width)
                     .zip_input<std::uint32_t>(1u)
                     .zip_input<std::uint32_t>(1u)
                     .branch([](auto first, auto second, auto alias, auto count,
                                auto ordinal) {
                       (void)alias;
                       (void)count;
                       (void)ordinal;
                       auto next_first =
                           first.map("window-alias-subview-first",
                                     [](auto value) { return value + 1u; });
                       auto next_second =
                           second.map("window-alias-subview-second",
                                      [](auto value) { return value + 2u; });
                       return outputs(next_first, next_second, next_first);
                     })
                     .compile();
  auto consumer =
      on(device)
          .input<std::uint32_t>(2u)
          .branch([](auto values) {
            return values.map("window-alias-subview-consumer",
                              [](auto value) { return value + 100u; });
          })
          .compile();
  auto first = device.upload<std::uint32_t>(first_seed);
  auto second = device.upload<std::uint32_t>(second_seed);
  auto count = device.upload<std::uint32_t>(count_value);
  auto backing = device.upload<std::uint32_t>(backing_seed);
  auto second_target = device.buffer<std::uint32_t>(width);
  auto sink = device.buffer<std::uint32_t>(2u);
  if (!program || !consumer || !first || !second || !count || !backing ||
      !second_target || !sink) {
    return 1;
  }
  auto target = backing->view(1u, width, 2u);
  auto downstream = backing->view(3u, 2u, 4u);
  if (!target || !downstream) {
    return 2;
  }
  const std::array<ResourceView, 3u> inputs{
      BufferAccess::view(*first, ResourceAccess::Read),
      BufferAccess::view(*second, ResourceAccess::Read),
      BufferAccess::view(*first, ResourceAccess::Read),
  };
  const std::array<ResourceView, 3u> outputs{
      BufferAccess::view(*target, ResourceAccess::Write),
      BufferAccess::view(*second_target, ResourceAccess::Write),
      BufferAccess::view(*target, ResourceAccess::Write),
  };
  auto build = make_pipeline(DeviceAccess::state(device));
  append_pipeline_windows(build, ProgramAccess::state(*program),
                          BufferAccess::view(*count, ResourceAccess::Read),
                          inputs, outputs, 4u, 1u, NoWindowTerminal, 1u);
  const std::array<ResourceView, 1u> consume_input{
      BufferAccess::view(*downstream, ResourceAccess::Read),
  };
  const std::array<ResourceView, 1u> consume_output{
      BufferAccess::view(*sink, ResourceAccess::Write),
  };
  append_pipeline(build, ProgramAccess::state(*consumer), consume_input,
                  consume_output);
  if (build == nullptr || build->failure != Reason::Ok ||
      build->steps.size() != 5u || build->internals.size() != 5u ||
      build->publications.size() != 2u) {
    return 3;
  }
  const auto plan = plan_pipeline(build);
  auto prepared = prepare_pipeline(build);
  if (!plan || !prepared || !run_pipeline(*prepared)) {
    return 4;
  }
  std::array<std::uint32_t, backing_seed.size()> backing_actual{};
  std::array<std::uint32_t, width> second_actual{};
  std::array<std::uint32_t, 2u> sink_actual{};
  if (!read_pipeline_raw(*prepared, BufferAccess::state(*backing), Type::U32,
                         FixedFormat{}, backing_actual.data(),
                         sizeof(backing_actual), backing_actual.size()) ||
      !read_pipeline_raw(*prepared, BufferAccess::state(*second_target),
                         Type::U32, FixedFormat{}, second_actual.data(),
                         sizeof(second_actual), second_actual.size()) ||
      !read_pipeline_raw(*prepared, BufferAccess::state(*sink), Type::U32,
                         FixedFormat{}, sink_actual.data(), sizeof(sink_actual),
                         sink_actual.size())) {
    return 5;
  }
  auto expected_backing = backing_seed;
  for (std::size_t index = 0u; index < width; ++index) {
    expected_backing[1u + 2u * index] = first_seed[index] + count_value[0u];
  }
  const std::array<std::uint32_t, width> expected_second{16u, 26u, 36u, 46u};
  const std::array<std::uint32_t, 2u> expected_sink{105u, 107u};
  return backing_actual == expected_backing &&
                 second_actual == expected_second &&
                 sink_actual == expected_sink
             ? 0
             : 6;
}

template <class Seed, class Fold>
[[nodiscard]] int CheckSealedPublicationMutation(Device &device,
                                                 const Seed &seed,
                                                 const Fold &fold) {
  using namespace rund::compute;
  using namespace rund::compute::detail;
  constexpr std::array<std::uint32_t, 1u> initial{kInitial};
  constexpr std::array<std::uint32_t, 1u> witness_values{0u};
  constexpr std::array<std::uint32_t, 3u> count_values{
      0x13579BDFu, static_cast<std::uint32_t>(kMaximum), 0x2468ACE0u};
  constexpr std::array<std::uint32_t, 1u> final_values{kSentinel};
  std::array<std::uint32_t, kWindowBacking> window_values{};
  window_values.fill(kSentinel);

  auto outer = device.upload<std::uint32_t>(initial);
  auto values = device.upload<std::uint32_t>(kValues);
  auto lanes = device.upload<std::uint32_t>(kLanes);
  auto witness = device.upload<std::uint32_t>(witness_values);
  auto count = device.upload<std::uint32_t>(count_values);
  auto final = device.upload<std::uint32_t>(final_values);
  auto appended = device.upload<std::uint32_t>(window_values);
  if (!outer || !values || !lanes || !witness || !count || !final ||
      !appended) {
    return 1;
  }
  auto count_view = count->view(1u, 1u);
  auto appended_view = appended->view(kWindowOffset, kMaximum);
  if (!count_view || !appended_view) {
    return 2;
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
      BufferAccess::view(*appended_view, ResourceAccess::Write),
  };
  auto build = make_pipeline(DeviceAccess::state(device));
  append_pipeline_window_repeat(
      build, ProgramAccess::state(seed), {}, ProgramAccess::state(fold),
      BufferAccess::view(*count_view, ResourceAccess::Read), inputs, finals,
      windows, kMaximum, kTile, 0u, NoWindowTerminal, 1u);
  for (std::size_t route = 1u; route < 3u; ++route) {
    auto drift = make_pipeline(DeviceAccess::state(device));
    append_pipeline_window_repeat(
        drift, ProgramAccess::state(seed), {}, ProgramAccess::state(fold),
        BufferAccess::view(*count_view, ResourceAccess::Read), inputs, finals,
        windows, kMaximum, kTile, 0u, NoWindowTerminal, 1u);
    if (drift == nullptr || drift->failure != Reason::Ok ||
        drift->nested_windows.size() != 1u) {
      return 3;
    }
    const PipelineBuildNestedWindow &nested = drift->nested_windows[0u];
    PipelineBinding &route_output =
        drift->steps[nested.shape.fold_first() + route]
            .outputs[nested.recurrent_output_count];
    route_output.owner = static_cast<std::uint32_t>(drift->internals.size());
    drift->internals.push_back(PipelineInternal{
        .type = route_output.type,
        .format = route_output.format,
        .count = route_output.count,
    });
    const auto invalid = plan_pipeline(drift);
    if (invalid || invalid.reason() != Reason::PipelineInvalid) {
      return static_cast<int>(9u + route);
    }
  }

  const auto plan = plan_pipeline(build);
  if (!plan || build == nullptr || build->memory == nullptr ||
      build->memory->publications.size() != 2u) {
    return 3;
  }
  auto authored = std::find_if(
      build->publications.begin(), build->publications.end(),
      [](const PipelineBuildPublication &publication) {
        return std::holds_alternative<PipelineBuildWindowPublication>(
            publication);
      });
  auto authored_terminal = std::find_if(
      build->publications.begin(), build->publications.end(),
      [](const PipelineBuildPublication &publication) {
        return std::holds_alternative<PipelineBuildTerminalPublication>(
            publication);
      });
  auto sealed = std::find_if(
      build->memory->publications.begin(), build->memory->publications.end(),
      [](const PipelinePublicationPlan &publication) {
        return std::holds_alternative<PipelineWindowPublicationPlan>(
            publication);
      });
  if (authored == build->publications.end() ||
      authored_terminal == build->publications.end() ||
      sealed == build->memory->publications.end()) {
    return 4;
  }
  auto &authored_window = std::get<PipelineBuildWindowPublication>(*authored);
  const auto &sealed_window = std::get<PipelineWindowPublicationPlan>(*sealed);
  if (build->nested_windows.size() != 1u ||
      build->memory->window_controls.size() != 1u) {
    return 5;
  }
  const PipelineWindowControl &sealed_control =
      build->memory->window_controls[0u];
  const std::uint64_t sealed_offset =
      sealed_window.target.view.identity.offset_bytes;
  if (sealed_offset != kWindowOffset * sizeof(std::uint32_t) ||
      sealed_control.count.identity.offset_bytes != sizeof(std::uint32_t) ||
      sealed_control.maximum != kMaximum || sealed_control.tile != kTile) {
    return 5;
  }

  // Mutate the cold authored publication and the sole authored control after
  // planning without invalidating memory. Preparation, accounting,
  // fingerprinting, and execution must retain the already sealed authority.
  authored_window.edge.target.offset = kWindowOffset + 1u;
  authored_window.edge.control = {};
  authored_window.edge.output.value = 0u;
  auto &authored_final =
      std::get<PipelineBuildTerminalPublication>(*authored_terminal);
  authored_final.edge.target.offset = 1u;
  authored_final.edge.control = {};
  authored_final.edge.output.value = 1u;
  if (build->window_controls.size() != 1u) {
    return 6;
  }
  PipelineBuildWindowControl &authored_control = build->window_controls[0u];
  authored_control.count_input = 0u;
  node::accel::detail::NestedTemplateShape drifted_shape{};
  if (!node::accel::detail::ProveNestedTemplateShape(
          build->nested_windows[0u].shape.first() + 1u,
          authored_control.maximum, authored_control.tile,
          build->nested_windows[0u].shape.inner_bound(), drifted_shape)) {
    return 6;
  }
  build->nested_windows[0u].shape = drifted_shape;
  authored_control.maximum = kTile;
  authored_control.tile = 1u;
  authored_control.terminal = 0u;
  authored_control.expected = 0xDEADBEEFu;
  auto prepared = prepare_pipeline(build);
  if (!prepared || (*prepared)->publications.size() != 2u) {
    return 6;
  }
  // Lock the complete v3 Pipeline identity, not only the isolated publication
  // suffix. In particular, nested routes retain zero/default ordinary-window
  // slots and serialize their control once in the nested-begin block.
  constexpr Fingerprint expected_nested{
      .hi = 0x99db298093fd0b2full,
      .lo = 0x8535483e8b5c9539ull,
  };
  if ((*prepared)->publication->fingerprint != expected_nested) {
    return 9;
  }
  const auto runtime = std::find_if(
      (*prepared)->publications.begin(), (*prepared)->publications.end(),
      [](const PipelinePublicationPlan &publication) {
        return std::holds_alternative<PipelineWindowPublicationPlan>(
            publication);
      });
  if (runtime == (*prepared)->publications.end() ||
      std::get<PipelineWindowPublicationPlan>(*runtime)
              .target.view.identity.offset_bytes != sealed_offset ||
      !run_pipeline(*prepared)) {
    return 7;
  }

  std::array<std::uint32_t, 1u> actual_final{};
  std::array<std::uint32_t, kWindowBacking> actual_window{};
  const Status final_read = read_pipeline_raw(
      *prepared, BufferAccess::state(*final), Type::U32, FixedFormat{},
      actual_final.data(), sizeof(actual_final), actual_final.size());
  const Status window_read = read_pipeline_raw(
      *prepared, BufferAccess::state(*appended), Type::U32, FixedFormat{},
      actual_window.data(), sizeof(actual_window), actual_window.size());
  std::array<std::uint32_t, kWindowBacking> expected_window{};
  expected_window.fill(kSentinel);
  const auto expected_payload = ExpectedWindow(kValues, kMaximum);
  std::copy(expected_payload.begin(), expected_payload.end(),
            expected_window.begin() + kWindowOffset);
  return final_read && window_read &&
                 actual_final[0u] == ExpectedFinal(kValues, kMaximum) &&
                 actual_window == expected_window
             ? 0
             : 8;
}

template <class Seed, class Fold>
[[nodiscard]] int CheckPublicationJobBindingMutation(Device &device,
                                                     const Seed &seed,
                                                     const Fold &fold) {
  using namespace rund::compute;
  using namespace rund::compute::detail;
  constexpr std::array<std::uint32_t, 1u> initial{kInitial};
  constexpr std::array<std::uint32_t, 1u> witness_values{0u};
  constexpr std::array<std::uint32_t, 1u> count_values{
      static_cast<std::uint32_t>(kMaximum)};
  constexpr std::array<std::uint32_t, 1u> final_values{kSentinel};
  std::array<std::uint32_t, kMaximum> window_values{};
  window_values.fill(kSentinel);

  auto outer = device.upload<std::uint32_t>(initial);
  auto values = device.upload<std::uint32_t>(kValues);
  auto lanes = device.upload<std::uint32_t>(kLanes);
  auto witness = device.upload<std::uint32_t>(witness_values);
  auto count = device.upload<std::uint32_t>(count_values);
  auto final = device.upload<std::uint32_t>(final_values);
  auto appended = device.upload<std::uint32_t>(window_values);
  auto second_count = device.upload<std::uint32_t>(count_values);
  auto ordinary_final_first = device.buffer<std::uint32_t>(1u);
  auto ordinary_tile_first = device.buffer<std::uint32_t>(kTile);
  auto ordinary_final_second = device.buffer<std::uint32_t>(1u);
  auto ordinary_tile_second = device.buffer<std::uint32_t>(kTile);
  if (!outer || !values || !lanes || !witness || !count || !final ||
      !appended || !second_count || !ordinary_final_first ||
      !ordinary_tile_first || !ordinary_final_second || !ordinary_tile_second) {
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
      BufferAccess::view(*appended, ResourceAccess::Write),
  };
  const std::array<ResourceView, 3u> ordinary_inputs{
      BufferAccess::view(*outer, ResourceAccess::Read),
      BufferAccess::view(*lanes, ResourceAccess::Read),
      BufferAccess::view(*witness, ResourceAccess::Read),
  };
  const std::array<ResourceView, 2u> ordinary_outputs_first{
      BufferAccess::view(*ordinary_final_first, ResourceAccess::Write),
      BufferAccess::view(*ordinary_tile_first, ResourceAccess::Write),
  };
  const std::array<ResourceView, 2u> ordinary_outputs_second{
      BufferAccess::view(*ordinary_final_second, ResourceAccess::Write),
      BufferAccess::view(*ordinary_tile_second, ResourceAccess::Write),
  };
  const auto make_build = [&] {
    auto build = make_pipeline(DeviceAccess::state(device));
    append_pipeline_window_repeat(
        build, ProgramAccess::state(seed), {}, ProgramAccess::state(fold),
        BufferAccess::view(*count, ResourceAccess::Read), inputs, finals,
        windows, kMaximum, kTile, 0u, NoWindowTerminal, 1u);
    return build;
  };

  auto bank_build = make_build();
  const auto bank_plan = plan_pipeline(bank_build);
  if (!bank_plan || bank_build == nullptr || bank_build->memory == nullptr ||
      bank_build->nested_windows.size() != 1u) {
    return 2;
  }
  const std::size_t fold_first =
      bank_build->nested_windows[0u].shape.fold_first();
  if (fold_first > bank_build->steps.size() ||
      bank_build->steps.size() - fold_first < 3u ||
      bank_build->steps[fold_first + 1u].outputs.empty() ||
      bank_build->steps[fold_first + 2u].outputs.empty()) {
    return 3;
  }
  const std::size_t seed_first =
      bank_build->nested_windows[0u].shape.seed_first();
  if (seed_first >= bank_build->steps.size() ||
      bank_build->steps[seed_first].inputs.size() < 2u) {
    return 4;
  }
  // The cached plan says route 2 writes the first recurrent bank. Redirect the
  // authored mirror to an already admitted owner without invalidating memory;
  // Job materialization must still consume the frozen step binding.
  PipelineBinding redirected =
      bank_build->steps[seed_first]
          .inputs[bank_build->steps[seed_first].inputs.size() - 2u];
  redirected.access = ResourceAccess::Write;
  redirected.hidden = true;
  bank_build->steps[fold_first + 2u].outputs[0u] = std::move(redirected);
  auto bank_prepared = prepare_pipeline(bank_build);
  if (!bank_prepared) {
    return 5;
  }

  auto count_build = make_build();
  const auto count_plan = plan_pipeline(count_build);
  if (!count_plan || count_build == nullptr || count_build->memory == nullptr ||
      count_build->nested_windows.size() != 1u) {
    return 6;
  }
  const PipelineBuildNestedWindow &nested = count_build->nested_windows[0u];
  if (nested.shape.seed_count() < 2u ||
      nested.shape.seed_first() >= count_build->steps.size() ||
      nested.shape.seed_first() + 1u >= count_build->steps.size()) {
    return 7;
  }
  PipelineBuildStep &later_seed =
      count_build->steps[nested.shape.seed_first() + 1u];
  if (later_seed.inputs.size() < 3u) {
    return 8;
  }
  // Replace the authored penultimate count input with the already admitted
  // one-element witness View. Every Seed Job must still use the frozen count
  // coordinate rather than this stale declaration.
  later_seed.inputs[later_seed.inputs.size() - 2u] = later_seed.inputs[2u];
  auto count_prepared = prepare_pipeline(count_build);
  if (!count_prepared) {
    return 9;
  }

  auto drift_build = make_pipeline(DeviceAccess::state(device));
  append_pipeline_windows(drift_build, ProgramAccess::state(fold),
                          BufferAccess::view(*count, ResourceAccess::Read),
                          ordinary_inputs, ordinary_outputs_first, kMaximum,
                          kTile, NoWindowTerminal, 1u);
  append_pipeline_windows(
      drift_build, ProgramAccess::state(fold),
      BufferAccess::view(*second_count, ResourceAccess::Read), ordinary_inputs,
      ordinary_outputs_second, kMaximum, kTile, NoWindowTerminal, 1u);
  if (drift_build == nullptr || drift_build->failure != Reason::Ok ||
      drift_build->window_controls.size() != 2u) {
    return 10;
  }
  const std::size_t drift_step =
      drift_build->window_controls[0u].ordinary_step.value + 1u;
  if (drift_step >= drift_build->steps.size()) {
    return 11;
  }
  drift_build->steps[drift_step].window_control = {.value = 1u};
  const auto drift_plan = plan_pipeline(drift_build);
  if (drift_plan || drift_plan.reason() != Reason::PipelineInvalid) {
    return 12;
  }
  drift_build->steps[drift_step].window_control = {};
  const auto unassigned_plan = plan_pipeline(drift_build);
  if (unassigned_plan || unassigned_plan.reason() != Reason::PipelineInvalid) {
    return 13;
  }

  for (const bool nested_first : {false, true}) {
    auto mixed_build = make_pipeline(DeviceAccess::state(device));
    const auto append_ordinary = [&] {
      append_pipeline_windows(
          mixed_build, ProgramAccess::state(fold),
          BufferAccess::view(*second_count, ResourceAccess::Read),
          ordinary_inputs, ordinary_outputs_second, kMaximum, kTile,
          NoWindowTerminal, 1u);
    };
    const auto append_nested = [&] {
      append_pipeline_window_repeat(
          mixed_build, ProgramAccess::state(seed), {},
          ProgramAccess::state(fold),
          BufferAccess::view(*count, ResourceAccess::Read), inputs, finals,
          windows, kMaximum, kTile, 0u, NoWindowTerminal, 1u);
    };
    if (nested_first) {
      append_nested();
      append_ordinary();
    } else {
      append_ordinary();
      append_nested();
    }
    const auto mixed_plan = plan_pipeline(mixed_build);
    if (!mixed_plan || mixed_build == nullptr ||
        mixed_build->memory == nullptr ||
        mixed_build->memory->window_controls.size() != 2u) {
      return 14;
    }
    auto mixed_prepared = prepare_pipeline(mixed_build);
    if (!mixed_prepared || !run_pipeline(*mixed_prepared)) {
      return 15;
    }
  }

  auto coordinate_build = make_build();
  const auto coordinate_plan = plan_pipeline(coordinate_build);
  if (!coordinate_plan || coordinate_build == nullptr ||
      coordinate_build->memory == nullptr ||
      coordinate_build->memory->window_controls.size() != 1u) {
    return 16;
  }
  auto mutable_plan =
      std::const_pointer_cast<PipelineMemoryPlan>(coordinate_build->memory);
  mutable_plan->window_controls[0u].count_input = 0u;
  auto coordinate_prepared = prepare_pipeline(coordinate_build);
  if (coordinate_prepared ||
      coordinate_prepared.reason() != Reason::PipelineInvalid) {
    return 17;
  }
  const auto verify = [&](const std::shared_ptr<PipelineState> &prepared) {
    std::array<std::uint32_t, 1u> actual_final{};
    std::array<std::uint32_t, kMaximum> actual_window{};
    const auto expected_window = ExpectedWindow(kValues, kMaximum);
    return run_pipeline(prepared) &&
           read_pipeline_raw(prepared, BufferAccess::state(*final), Type::U32,
                             FixedFormat{}, actual_final.data(),
                             sizeof(actual_final), actual_final.size()) &&
           read_pipeline_raw(prepared, BufferAccess::state(*appended),
                             Type::U32, FixedFormat{}, actual_window.data(),
                             sizeof(actual_window), actual_window.size()) &&
           actual_final[0u] == ExpectedFinal(kValues, kMaximum) &&
           actual_window == expected_window;
  };
  return verify(*bank_prepared) && verify(*count_prepared) ? 0 : 18;
}

template <class Seed, class Fold, class Advance>
[[nodiscard]] int
CheckTransactionalCountParity(Device &device, const Seed &seed,
                              const Fold &fold, const Advance &advance) {
  using namespace rund::compute;
  using namespace rund::compute::detail;
  constexpr std::array<std::uint32_t, 1u> initial{kInitial};
  constexpr std::array<std::uint32_t, 1u> witness_values{0u};
  constexpr std::array<std::uint32_t, 1u> first_count{
      static_cast<std::uint32_t>(kTile)};
  constexpr std::array<std::uint32_t, 1u> second_count{0u};
  constexpr std::array<std::uint32_t, 1u> final_values{kSentinel};
  std::array<std::uint32_t, kMaximum> window_values{};
  window_values.fill(kSentinel);

  auto outer = device.upload<std::uint32_t>(initial);
  auto values = device.upload<std::uint32_t>(kValues);
  auto lanes = device.upload<std::uint32_t>(kLanes);
  auto witness = device.upload<std::uint32_t>(witness_values);
  auto count_first = device.upload<std::uint32_t>(first_count);
  auto count_second = device.upload<std::uint32_t>(second_count);
  auto final = device.upload<std::uint32_t>(final_values);
  auto appended = device.upload<std::uint32_t>(window_values);
  if (!outer || !values || !lanes || !witness || !count_first ||
      !count_second || !final || !appended) {
    return 1;
  }

  const auto body = tile_repeat<0u>(seed, fold);
  auto builder = pipeline(device);
  builder.state(*count_first, *count_second)
      .then(advance, read(*count_first), write(*count_second))
      .template windows<kMaximum, kTile>(
          body, rund::compute::window(*count_second),
          read(*outer, *values, *lanes, *witness), write_final(*final),
          write_window(*appended))
      .commit();
  auto prepared = std::move(builder).prepare();
  if (!prepared) {
    return 2;
  }
  const std::shared_ptr<PipelineState> &state =
      PipelineStateAccess::state(*prepared);
  const auto publication = std::find_if(
      state->publications.begin(), state->publications.end(),
      [](const PipelinePublicationPlan &value) {
        return std::holds_alternative<PipelineWindowPublicationPlan>(value);
      });
  if (state == nullptr || publication == state->publications.end()) {
    return 3;
  }
  const auto &window = std::get<PipelineWindowPublicationPlan>(*publication);
  const std::uint32_t count_ordinal =
      state->windows[window.state].control.count.identity.resource_ordinal;
  if (count_ordinal >= state->resources.size() ||
      state->resources[count_ordinal].partner >= state->resources.size() ||
      state->resources[state->resources[count_ordinal].partner].partner !=
          count_ordinal) {
    return 4;
  }

  const auto check = [&](const std::uint32_t count_value) {
    const Status ran = prepared->run();
    std::array<std::uint32_t, 1u> actual_final{};
    std::array<std::uint32_t, kMaximum> actual_window{};
    const auto expected_window = ExpectedWindow(kValues, count_value);
    return ran && prepared->read(*final, actual_final) &&
           prepared->read(*appended, actual_window) &&
           actual_final[0u] == ExpectedFinal(kValues, count_value) &&
           actual_window == expected_window;
  };
  if (!check(static_cast<std::uint32_t>(2u * kTile))) {
    return 5;
  }
  if (!check(static_cast<std::uint32_t>(3u * kTile)) ||
      prepared->generation() != 2u) {
    return 6;
  }
  return 0;
}

template <class Seed, class Fold>
[[nodiscard]] int CheckStatePairPublicationTargetRejected(Device &device,
                                                          const Seed &seed,
                                                          const Fold &fold) {
  using namespace rund::compute;
  constexpr std::array<std::uint32_t, 1u> initial{kInitial};
  constexpr std::array<std::uint32_t, 1u> witness_values{0u};
  constexpr std::array<std::uint32_t, 1u> count_values{
      static_cast<std::uint32_t>(kMaximum)};
  constexpr std::array<std::uint32_t, 1u> pending_values{kSentinel};
  std::array<std::uint32_t, kMaximum> window_values{};
  window_values.fill(kSentinel);

  auto published = device.upload<std::uint32_t>(initial);
  auto values = device.upload<std::uint32_t>(kValues);
  auto lanes = device.upload<std::uint32_t>(kLanes);
  auto witness = device.upload<std::uint32_t>(witness_values);
  auto count = device.upload<std::uint32_t>(count_values);
  auto pending = device.upload<std::uint32_t>(pending_values);
  auto appended = device.upload<std::uint32_t>(window_values);
  if (!published || !values || !lanes || !witness || !count || !pending ||
      !appended) {
    return 1;
  }

  const auto body = tile_repeat<0u>(seed, fold);
  auto builder = pipeline(device);
  builder.state(*published, *pending)
      .template windows<kMaximum, kTile>(
          body, rund::compute::window(*count),
          read(*published, *values, *lanes, *witness), write_final(*pending),
          write_window(*appended))
      .commit();
  const auto plan = builder.plan();
  auto prepared = std::move(builder).prepare();
  return plan && !prepared && prepared.reason() == Reason::PipelineInvalid ? 0
                                                                           : 2;
}

[[nodiscard]] bool PreparedShape(const rund::compute::Pipeline &pipeline) {
  using namespace rund::compute::detail;
  const std::shared_ptr<PipelineState> &state =
      PipelineStateAccess::state(pipeline);
  if (state == nullptr || state->steps.size() != kTemplates ||
      state->windows.size() != 1u || state->publications.size() != 2u) {
    return false;
  }
  const PipelineWindow &window = state->windows[0u];
  const auto &shape = window.nested_shape;
  if (!window.nested() || shape.first() != 0u || shape.end() != kTemplates ||
      shape.seed_first() != 0u || shape.seed_count() != kOuter ||
      shape.action_first() != kOuter || shape.action_count() != 0u ||
      shape.fold_first() != kOuter || window.recurrent_output_count != 1u ||
      window.control.maximum != kMaximum || window.control.tile != kTile) {
    return false;
  }
  for (std::size_t index = 0u; index < state->steps.size(); ++index) {
    const PipelineRoute expected =
        index < kOuter ? PipelineRoute::NestedSeed : PipelineRoute::NestedFold;
    if (state->steps[index].route != expected ||
        state->steps[index].job == nullptr) {
      return false;
    }
  }
  const auto terminal = std::find_if(
      state->publications.begin(), state->publications.end(),
      [](const PipelinePublicationPlan &value) {
        return std::holds_alternative<PipelineTerminalPublicationPlan>(value);
      });
  const auto appended = std::find_if(
      state->publications.begin(), state->publications.end(),
      [](const PipelinePublicationPlan &value) {
        return std::holds_alternative<PipelineWindowPublicationPlan>(value);
      });
  if (terminal == state->publications.end() ||
      appended == state->publications.end()) {
    return false;
  }
  const auto &terminal_plan =
      std::get<PipelineTerminalPublicationPlan>(*terminal);
  const auto &window_plan = std::get<PipelineWindowPublicationPlan>(*appended);
  return window.control.final < terminal_plan.sources.size() &&
         terminal_plan.sources[window.control.final].identity.count == 1u &&
         window_plan.source.identity.count == kTile &&
         window.control.count.identity.count == 1u &&
         window_plan.target.view.identity.stride_bytes ==
             window_plan.target.view.identity.element_bytes;
}

template <class Seed, class Fold>
[[nodiscard]] int
CheckCount(Device &device, const Backend backend, const Seed &seed,
           const Fold &fold, const std::uint32_t count_value,
           const std::array<std::uint32_t, kMaximum> &values,
           const bool high_index_only, WindowOutputIdentity &identity) {
  using namespace rund::compute;
  constexpr std::array<std::uint32_t, 1u> initial{kInitial};
  constexpr std::array<std::uint32_t, 1u> witness_values{0u};
  const std::array<std::uint32_t, 3u> count_values{0x13579BDFu, count_value,
                                                   0x2468ACE0u};
  constexpr std::array<std::uint32_t, 1u> final_values{kSentinel};
  std::array<std::uint32_t, kWindowBacking> window_values{};
  window_values.fill(kSentinel);

  auto outer = device.upload<std::uint32_t>(initial);
  auto input_values = device.upload<std::uint32_t>(values);
  auto lanes = device.upload<std::uint32_t>(kLanes);
  auto witness = device.upload<std::uint32_t>(witness_values);
  auto count = device.upload<std::uint32_t>(count_values);
  auto final = device.upload<std::uint32_t>(final_values);
  auto appended = device.upload<std::uint32_t>(window_values);
  if (!outer || !input_values || !lanes || !witness || !count || !final ||
      !appended) {
    return 1;
  }
  auto count_view = count->view(1u, 1u);
  auto appended_view = appended->view(kWindowOffset, kMaximum);
  if (!count_view || !appended_view) {
    return 2;
  }

  const auto body = tile_repeat<0u>(seed, fold);
  auto builder = pipeline(device);
  builder.windows<kMaximum, kTile>(
      body, rund::compute::window(*count_view),
      read(*outer, *input_values, *lanes, *witness), write_final(*final),
      write_window(*appended_view));
  const auto plan = builder.plan();
  if (!plan || plan->outer_window_count != kOuter ||
      plan->tile_capacity != kTile || plan->inner_iteration_count != 0u ||
      plan->prepared_template_count != kTemplates ||
      plan->prepared_command_count != kCommands ||
      plan->publish_count != kOuter + 1u ||
      plan->publish_bytes != (kMaximum + 1u) * sizeof(std::uint32_t)) {
    if (!plan) {
      std::fprintf(stderr, "window output plan backend=%u count=%u reason=%u\n",
                   static_cast<unsigned>(backend), count_value,
                   static_cast<unsigned>(plan.reason()));
    } else {
      std::fprintf(
          stderr,
          "window output plan backend=%u count=%u outer=%llu "
          "tile=%llu inner=%llu templates=%llu commands=%llu "
          "publish=%llu/%llu state=%llu transient=%llu\n",
          static_cast<unsigned>(backend), count_value,
          static_cast<unsigned long long>(plan->outer_window_count),
          static_cast<unsigned long long>(plan->tile_capacity),
          static_cast<unsigned long long>(plan->inner_iteration_count),
          static_cast<unsigned long long>(plan->prepared_template_count),
          static_cast<unsigned long long>(plan->prepared_command_count),
          static_cast<unsigned long long>(plan->publish_count),
          static_cast<unsigned long long>(plan->publish_bytes),
          static_cast<unsigned long long>(plan->state_bytes),
          static_cast<unsigned long long>(plan->transient_bytes));
    }
    return 3;
  }
  auto prepared = std::move(builder)
                      .budget(MemoryBudget{.bytes = plan->peak_bytes})
                      .prepare();
  if (!prepared || prepared->plan() != *plan || !PreparedShape(*prepared)) {
    std::fprintf(stderr,
                 "window output prepare backend=%u count=%u ok=%u reason=%u\n",
                 static_cast<unsigned>(backend), count_value,
                 static_cast<unsigned>(prepared.ok()),
                 static_cast<unsigned>(prepared.reason()));
    return 4;
  }

  std::array<std::uint32_t, 1u> first_final{};
  std::array<std::uint32_t, kMaximum> first_window{};
  std::array<std::uint32_t, kWindowBacking> first_backing{};
  const Status first_run = prepared->run();
  const Status first_final_read =
      first_run ? prepared->read(*final, first_final) : first_run;
  const Status first_window_read =
      first_final_read ? prepared->read(*appended, first_backing)
                       : first_final_read;
  if (!first_run || !first_final_read || !first_window_read) {
    std::fprintf(stderr,
                 "window output first run backend=%u count=%u high=%u "
                 "run=%u/%u final=%u/%u window=%u/%u\n",
                 static_cast<unsigned>(backend), count_value,
                 static_cast<unsigned>(high_index_only),
                 static_cast<unsigned>(first_run.ok()),
                 static_cast<unsigned>(first_run.reason()),
                 static_cast<unsigned>(first_final_read.ok()),
                 static_cast<unsigned>(first_final_read.reason()),
                 static_cast<unsigned>(first_window_read.ok()),
                 static_cast<unsigned>(first_window_read.reason()));
    return 5;
  }
  std::copy_n(first_backing.begin() + kWindowOffset, kMaximum,
              first_window.begin());
  const MemoryStats after_observation = prepared->memory();
  const Status second = prepared->run();
  const Stats stats = prepared->stats();
  const MemoryStats after_warm = prepared->memory();
  std::array<std::uint32_t, 1u> actual_final{};
  std::array<std::uint32_t, kMaximum> actual_window{};
  std::array<std::uint32_t, kWindowBacking> actual_backing{};
  const Status final_read = prepared->read(*final, actual_final);
  const Status window_read = prepared->read(*appended, actual_backing);
  if (window_read) {
    std::copy_n(actual_backing.begin() + kWindowOffset, kMaximum,
                actual_window.begin());
  }
  const std::size_t active = CeilDiv(count_value, kTile);
  const auto expected_window = high_index_only
                                   ? ExpectedHighIndexWindow()
                                   : ExpectedWindow(values, count_value);
  const bool stats_match =
      stats.command_submits == (backend == Backend::Cpu ? 0u : 1u) &&
      stats.pipeline.step_count == 1u &&
      stats.pipeline.verified_step_count == 1u &&
      stats.pipeline.failed_step_index == PipelineStats::no_failed_step &&
      stats.pipeline.executed_outer_window_count == active &&
      stats.pipeline.skipped_outer_window_count == kOuter - active &&
      stats.pipeline.executed_inner_iteration_count == 0u &&
      stats.pipeline.skipped_inner_iteration_count == 0u &&
      stats.pipeline.prepared_template_count == kTemplates &&
      stats.pipeline.prepared_command_count == kCommands &&
      stats.publication.generation == 2u &&
      stats.publication.discard_count == 0u && WarmSetupClean(stats);
  if (!second || prepared->generation() != 2u || !stats_match ||
      !NoAllocation(after_observation, after_warm) || !final_read ||
      !window_read || first_final != actual_final ||
      first_window != actual_window ||
      actual_final[0u] != ExpectedFinal(values, count_value) ||
      actual_window != expected_window ||
      (high_index_only &&
       (!std::all_of(
            actual_window.begin(), actual_window.end() - 1u,
            [](const std::uint32_t value) { return value == kSentinel; }) ||
        actual_window.back() != kHighIndexValue)) ||
      !std::all_of(
          actual_backing.begin(), actual_backing.begin() + kWindowOffset,
          [](const std::uint32_t value) { return value == kSentinel; }) ||
      !std::all_of(actual_backing.begin() + kWindowOffset + kMaximum,
                   actual_backing.end(), [](const std::uint32_t value) {
                     return value == kSentinel;
                   })) {
    std::fprintf(
        stderr,
        "window output run backend=%u count=%u status=%u/%u generation=%llu "
        "final=%u/%u first=%u/%u last=%u/%u outer=%llu/%llu "
        "skipped=%llu/%llu inner=%llu/%llu templates=%llu commands=%llu "
        "rebind=%llu alloc=%llu submit=%llu\n",
        static_cast<unsigned>(backend), count_value,
        static_cast<unsigned>(second.ok()),
        static_cast<unsigned>(second.reason()),
        static_cast<unsigned long long>(prepared->generation()),
        actual_final[0u], ExpectedFinal(values, count_value), actual_window[0u],
        expected_window[0u], actual_window[kMaximum - 1u],
        expected_window[kMaximum - 1u],
        static_cast<unsigned long long>(
            stats.pipeline.executed_outer_window_count),
        static_cast<unsigned long long>(active),
        static_cast<unsigned long long>(
            stats.pipeline.skipped_outer_window_count),
        static_cast<unsigned long long>(kOuter - active),
        static_cast<unsigned long long>(
            stats.pipeline.executed_inner_iteration_count),
        static_cast<unsigned long long>(
            stats.pipeline.skipped_inner_iteration_count),
        static_cast<unsigned long long>(stats.pipeline.prepared_template_count),
        static_cast<unsigned long long>(stats.pipeline.prepared_command_count),
        static_cast<unsigned long long>(stats.pipeline.rebinding_count),
        static_cast<unsigned long long>(stats.buffer_allocations),
        static_cast<unsigned long long>(stats.command_submits));
    return 6;
  }

  if (count_value == kMaximum) {
    std::array<std::uint32_t, kMaximum + 1u> identity_values{};
    identity_values[0u] = actual_final[0u];
    std::copy(actual_window.begin(), actual_window.end(),
              identity_values.begin() + 1u);
    const Fingerprint current_seed = seed.fingerprint();
    const Fingerprint current_fold = fold.fingerprint();
    const Fingerprint current_pipeline = prepared->fingerprint();
    const std::uint64_t current_output =
        Hash(identity_values.data(), sizeof(identity_values));
    if (!current_seed || !current_fold || !current_pipeline ||
        current_output == 0u) {
      return 7;
    }
    if (identity.seed) {
      if (identity.seed != current_seed) {
        return 8;
      }
    } else {
      identity.seed = current_seed;
    }
    Fingerprint &fold_identity =
        high_index_only ? identity.high_index_fold : identity.fold;
    Fingerprint &pipeline_identity =
        high_index_only ? identity.high_index_pipeline : identity.pipeline;
    std::uint64_t &output_identity =
        high_index_only ? identity.high_index_output : identity.output;
    if (fold_identity) {
      if (fold_identity != current_fold ||
          pipeline_identity != current_pipeline ||
          output_identity != current_output) {
        return 8;
      }
    } else {
      fold_identity = current_fold;
      pipeline_identity = current_pipeline;
      output_identity = current_output;
    }
  }
  return 0;
}

template <class Seed, class Fold, class Downstream>
[[nodiscard]] int CheckDownstreamRead(Device &device, const Backend backend,
                                      const Seed &seed, const Fold &fold,
                                      const Downstream &downstream,
                                      WindowOutputIdentity &identity) {
  using namespace rund::compute;
  constexpr std::uint32_t count_value = 5u;
  constexpr std::array<std::uint32_t, 1u> initial{kInitial};
  constexpr std::array<std::uint32_t, 1u> witness_values{0u};
  constexpr std::array<std::uint32_t, 1u> count_values{count_value};
  constexpr std::array<std::uint32_t, 1u> final_values{kSentinel};
  std::array<std::uint32_t, kMaximum> window_values{};
  window_values.fill(kSentinel);

  auto outer = device.upload<std::uint32_t>(initial);
  auto values = device.upload<std::uint32_t>(kValues);
  auto lanes = device.upload<std::uint32_t>(kLanes);
  auto witness = device.upload<std::uint32_t>(witness_values);
  auto count = device.upload<std::uint32_t>(count_values);
  auto final = device.upload<std::uint32_t>(final_values);
  auto appended = device.upload<std::uint32_t>(window_values);
  auto sink = device.upload<std::uint32_t>(window_values);
  if (!outer || !values || !lanes || !witness || !count || !final ||
      !appended || !sink) {
    return 1;
  }

  const auto body = tile_repeat<0u>(seed, fold);
  auto baseline = pipeline(device);
  baseline.windows<kMaximum, kTile>(body, rund::compute::window(*count),
                                    read(*outer, *values, *lanes, *witness),
                                    write_final(*final),
                                    write_window(*appended));
  const auto baseline_plan = baseline.plan();

  auto builder = pipeline(device);
  builder
      .windows<kMaximum, kTile>(body, rund::compute::window(*count),
                                read(*outer, *values, *lanes, *witness),
                                write_final(*final), write_window(*appended))
      .then(downstream, read(*appended), write(*sink));
  const auto plan = builder.plan();
  if (!baseline_plan || !plan ||
      plan->prepared_template_count !=
          baseline_plan->prepared_template_count + 1u ||
      plan->prepared_command_count !=
          baseline_plan->prepared_command_count + 1u ||
      plan->barrier_count != baseline_plan->barrier_count + 1u ||
      plan->resource_count != baseline_plan->resource_count + 1u ||
      plan->persistent_bytes !=
          baseline_plan->persistent_bytes + kMaximum * sizeof(std::uint32_t) ||
      plan->state_bytes != baseline_plan->state_bytes ||
      plan->publish_count != baseline_plan->publish_count ||
      plan->publish_bytes != baseline_plan->publish_bytes) {
    std::fprintf(
        stderr,
        "window output downstream plan backend=%u status=%u/%u reason=%u/%u "
        "templates=%llu/%llu commands=%llu/%llu barriers=%llu/%llu "
        "resources=%llu/%llu persistent=%llu/%llu state=%llu/%llu "
        "publish=%llu/%llu/%llu/%llu\n",
        static_cast<unsigned>(backend),
        static_cast<unsigned>(baseline_plan.ok()),
        static_cast<unsigned>(plan.ok()),
        static_cast<unsigned>(baseline_plan.reason()),
        static_cast<unsigned>(plan.reason()),
        static_cast<unsigned long long>(
            baseline_plan ? baseline_plan->prepared_template_count : 0u),
        static_cast<unsigned long long>(plan ? plan->prepared_template_count
                                             : 0u),
        static_cast<unsigned long long>(
            baseline_plan ? baseline_plan->prepared_command_count : 0u),
        static_cast<unsigned long long>(plan ? plan->prepared_command_count
                                             : 0u),
        static_cast<unsigned long long>(
            baseline_plan ? baseline_plan->barrier_count : 0u),
        static_cast<unsigned long long>(plan ? plan->barrier_count : 0u),
        static_cast<unsigned long long>(
            baseline_plan ? baseline_plan->resource_count : 0u),
        static_cast<unsigned long long>(plan ? plan->resource_count : 0u),
        static_cast<unsigned long long>(
            baseline_plan ? baseline_plan->persistent_bytes : 0u),
        static_cast<unsigned long long>(plan ? plan->persistent_bytes : 0u),
        static_cast<unsigned long long>(
            baseline_plan ? baseline_plan->state_bytes : 0u),
        static_cast<unsigned long long>(plan ? plan->state_bytes : 0u),
        static_cast<unsigned long long>(
            baseline_plan ? baseline_plan->publish_count : 0u),
        static_cast<unsigned long long>(plan ? plan->publish_count : 0u),
        static_cast<unsigned long long>(
            baseline_plan ? baseline_plan->publish_bytes : 0u),
        static_cast<unsigned long long>(plan ? plan->publish_bytes : 0u));
    return 2;
  }

  auto prepared = std::move(builder)
                      .budget(MemoryBudget{.bytes = plan->peak_bytes})
                      .prepare();
  if (!prepared) {
    return 3;
  }
  using namespace rund::compute::detail;
  const std::shared_ptr<PipelineState> &state =
      PipelineStateAccess::state(*prepared);
  if (state == nullptr || state->windows.size() != 1u ||
      state->steps.size() != kTemplates + 1u ||
      state->windows[0u].recurrent_output_count != 1u ||
      state->barriers.back() == 0u) {
    return 4;
  }
  const auto publication = std::find_if(
      state->publications.begin(), state->publications.end(),
      [](const PipelinePublicationPlan &value) {
        return std::holds_alternative<PipelineWindowPublicationPlan>(value);
      });
  if (publication == state->publications.end()) {
    return 5;
  }
  const auto &window_publication =
      std::get<PipelineWindowPublicationPlan>(*publication);
  const std::uint32_t resource_index =
      window_publication.target.view.identity.resource_ordinal;
  if (resource_index >= state->resources.size()) {
    return 6;
  }
  const PipelineWindow &window = state->windows[0u];
  const bool has_publication_read_hazard = std::any_of(
      state->dependencies.begin(), state->dependencies.end(),
      [&](const PipelineDependency &dependency) {
        return dependency.resource == resource_index &&
               dependency.before >= window.nested_shape.fold_first() &&
               dependency.before < window.nested_shape.end() &&
               dependency.after == kTemplates &&
               dependency.before_access == PipelineAccess::Write &&
               dependency.after_access == PipelineAccess::Read;
      });
  if (!has_publication_read_hazard) {
    return 7;
  }

  std::array<std::uint32_t, 1u> first_final{};
  std::array<std::uint32_t, kMaximum> first_window{};
  std::array<std::uint32_t, kMaximum> first_sink{};
  if (!prepared->run() || !prepared->read(*final, first_final) ||
      !prepared->read(*appended, first_window) ||
      !prepared->read(*sink, first_sink)) {
    return 8;
  }
  const MemoryStats before_warm = prepared->memory();
  const Status second = prepared->run();
  const Stats stats = prepared->stats();
  const MemoryStats after_warm = prepared->memory();
  std::array<std::uint32_t, 1u> actual_final{};
  std::array<std::uint32_t, kMaximum> actual_window{};
  std::array<std::uint32_t, kMaximum> actual_sink{};
  const Status final_read = prepared->read(*final, actual_final);
  const Status window_read = prepared->read(*appended, actual_window);
  const Status sink_read = prepared->read(*sink, actual_sink);
  const auto expected_window = ExpectedWindow(kValues, count_value);
  if (!second || !final_read || !window_read || !sink_read ||
      prepared->generation() != 2u ||
      actual_final[0u] != ExpectedFinal(kValues, count_value) ||
      actual_window != expected_window || actual_sink != expected_window ||
      first_final != actual_final || first_window != actual_window ||
      first_sink != actual_sink || !NoAllocation(before_warm, after_warm) ||
      !WarmSetupClean(stats) || stats.pipeline.step_count != 2u ||
      stats.pipeline.verified_step_count != 2u ||
      stats.pipeline.barrier_count != plan->barrier_count ||
      stats.pipeline.executed_outer_window_count != 2u ||
      stats.pipeline.skipped_outer_window_count != kOuter - 2u ||
      stats.command_submits != (backend == Backend::Cpu ? 0u : 1u)) {
    std::fprintf(
        stderr,
        "window output downstream run backend=%u status=%u/%u generation=%llu "
        "final=%u/%u window=%u/%u sink=%u/%u steps=%llu/%llu "
        "barriers=%llu/%llu outer=%llu/%llu skipped=%llu/%llu submit=%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(second.ok()),
        static_cast<unsigned>(second.reason()),
        static_cast<unsigned long long>(prepared->generation()),
        actual_final[0u], ExpectedFinal(kValues, count_value),
        actual_window[0u], expected_window[0u], actual_sink[0u],
        expected_window[0u],
        static_cast<unsigned long long>(stats.pipeline.step_count), 2ull,
        static_cast<unsigned long long>(stats.pipeline.barrier_count),
        static_cast<unsigned long long>(plan->barrier_count),
        static_cast<unsigned long long>(
            stats.pipeline.executed_outer_window_count),
        2ull,
        static_cast<unsigned long long>(
            stats.pipeline.skipped_outer_window_count),
        static_cast<unsigned long long>(kOuter - 2u),
        static_cast<unsigned long long>(stats.command_submits));
    return 9;
  }

  std::array<std::uint32_t, 2u * kMaximum + 1u> raw{};
  raw[0u] = actual_final[0u];
  std::copy(actual_window.begin(), actual_window.end(), raw.begin() + 1u);
  std::copy(actual_sink.begin(), actual_sink.end(),
            raw.begin() + 1u + kMaximum);
  const Fingerprint current_program = downstream.fingerprint();
  const Fingerprint current_pipeline = prepared->fingerprint();
  const std::uint64_t current_output = Hash(raw.data(), sizeof(raw));
  if (!current_program || !current_pipeline || current_output == 0u) {
    return 10;
  }
  if (identity.downstream_program) {
    if (identity.downstream_program != current_program ||
        identity.downstream_pipeline != current_pipeline ||
        identity.downstream_output != current_output) {
      return 11;
    }
  } else {
    identity.downstream_program = current_program;
    identity.downstream_pipeline = current_pipeline;
    identity.downstream_output = current_output;
  }

  auto write_conflict = pipeline(device);
  write_conflict
      .windows<kMaximum, kTile>(body, rund::compute::window(*count),
                                read(*outer, *values, *lanes, *witness),
                                write_final(*final), write_window(*appended))
      .then(downstream, read(*appended), write(*appended));
  const auto conflict_plan = write_conflict.plan();
  if (conflict_plan ||
      conflict_plan.reason() != Reason::BindingAliasUnsupported) {
    return 12;
  }
  return 0;
}

template <class Seed, class Fold>
[[nodiscard]] int CheckZeroRecurrentPrefix(Device &device,
                                           const Backend backend,
                                           const Seed &seed, const Fold &fold,
                                           WindowOutputIdentity &identity) {
  using namespace rund::compute;
  constexpr std::array<std::uint32_t, 1u> initial{kInitial};
  constexpr std::array<std::uint32_t, 1u> count_values{0u};
  constexpr std::array<std::uint32_t, kTile> tile_values{3u, 5u, 7u, 11u};
  constexpr std::array<std::uint32_t, 1u> final_values{kSentinel};
  std::array<std::uint32_t, kMaximum> window_values{};
  window_values.fill(kSentinel);

  auto outer = device.upload<std::uint32_t>(initial);
  auto tile = device.upload<std::uint32_t>(tile_values);
  auto count = device.upload<std::uint32_t>(count_values);
  auto final = device.upload<std::uint32_t>(final_values);
  auto appended = device.upload<std::uint32_t>(window_values);
  if (!outer || !tile || !count || !final || !appended) {
    return 1;
  }

  const auto body = tile_repeat<0u>(seed, fold);
  auto builder = pipeline(device);
  builder.windows<kMaximum, kTile>(body, rund::compute::window(*count),
                                   read(*outer, *tile), write_final(*final),
                                   write_window(*appended));
  const auto plan = builder.plan();
  if (!plan || plan->prepared_template_count != kTemplates ||
      plan->prepared_command_count != kCommands ||
      plan->publish_count != kOuter + 1u ||
      plan->publish_bytes != (kMaximum + 1u) * sizeof(std::uint32_t)) {
    return 2;
  }
  auto prepared = std::move(builder)
                      .budget(MemoryBudget{.bytes = plan->peak_bytes})
                      .prepare();
  if (!prepared) {
    return 3;
  }
  using namespace rund::compute::detail;
  const std::shared_ptr<PipelineState> &state =
      PipelineStateAccess::state(*prepared);
  if (state == nullptr || state->windows.size() != 1u ||
      state->windows[0u].recurrent_output_count != 1u ||
      state->steps[state->windows[0u].nested_shape.fold_first()].job ==
          nullptr) {
    return 4;
  }

  std::array<std::uint32_t, 1u> first_final{};
  std::array<std::uint32_t, kMaximum> first_window{};
  if (!prepared->run() || !prepared->read(*final, first_final) ||
      !prepared->read(*appended, first_window)) {
    return 5;
  }
  const MemoryStats before_warm = prepared->memory();
  const Status second = prepared->run();
  const Stats stats = prepared->stats();
  const MemoryStats after_warm = prepared->memory();
  std::array<std::uint32_t, 1u> actual_final{};
  std::array<std::uint32_t, kMaximum> actual_window{};
  const Status final_read = prepared->read(*final, actual_final);
  const Status window_read = prepared->read(*appended, actual_window);
  if (!second || !final_read || !window_read || prepared->generation() != 2u ||
      actual_final != initial || actual_window != window_values ||
      first_final != actual_final || first_window != actual_window ||
      !NoAllocation(before_warm, after_warm) || !WarmSetupClean(stats) ||
      stats.pipeline.step_count != 1u ||
      stats.pipeline.verified_step_count != 1u ||
      stats.pipeline.executed_outer_window_count != 0u ||
      stats.pipeline.skipped_outer_window_count != kOuter ||
      stats.pipeline.executed_inner_iteration_count != 0u ||
      stats.pipeline.skipped_inner_iteration_count != 0u ||
      stats.command_submits != (backend == Backend::Cpu ? 0u : 1u)) {
    std::fprintf(
        stderr,
        "window output zero prefix backend=%u status=%u/%u generation=%llu "
        "final=%u/%u window=%u/%u outer=%llu skipped=%llu/%llu submit=%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(second.ok()),
        static_cast<unsigned>(second.reason()),
        static_cast<unsigned long long>(prepared->generation()),
        actual_final[0u], initial[0u], actual_window[0u], window_values[0u],
        static_cast<unsigned long long>(
            stats.pipeline.executed_outer_window_count),
        static_cast<unsigned long long>(
            stats.pipeline.skipped_outer_window_count),
        static_cast<unsigned long long>(kOuter),
        static_cast<unsigned long long>(stats.command_submits));
    return 6;
  }

  std::array<std::uint32_t, kMaximum + 1u> raw{};
  raw[0u] = actual_final[0u];
  std::copy(actual_window.begin(), actual_window.end(), raw.begin() + 1u);
  const Fingerprint current_seed = seed.fingerprint();
  const Fingerprint current_fold = fold.fingerprint();
  const Fingerprint current_pipeline = prepared->fingerprint();
  const std::uint64_t current_output = Hash(raw.data(), sizeof(raw));
  if (!current_seed || !current_fold || !current_pipeline ||
      current_output == 0u) {
    return 7;
  }
  if (identity.zero_seed) {
    if (identity.zero_seed != current_seed ||
        identity.zero_fold != current_fold ||
        identity.zero_pipeline != current_pipeline ||
        identity.zero_output != current_output) {
      return 8;
    }
  } else {
    identity.zero_seed = current_seed;
    identity.zero_fold = current_fold;
    identity.zero_pipeline = current_pipeline;
    identity.zero_output = current_output;
  }
  return 0;
}

template <class Seed, class Fold>
[[nodiscard]] int CheckScatterConflicts(Device &device, const Backend backend,
                                        const Seed &seed, const Fold &fold,
                                        WindowOutputIdentity &identity) {
  using namespace rund::compute;
  constexpr std::array<std::array<std::uint32_t, kTile>, 2u> values{{
      {5u, 5u, 0u, 0u},
      {5u, 7u, 11u, 13u},
  }};
  constexpr std::array<std::array<std::uint32_t, kTile>, 2u> targets{{
      {3u, 3u, 0u, 1u},
      {3u, 3u, 3u, 3u},
  }};
  constexpr std::array<std::uint32_t, 2u> expected{10u, 36u};
  constexpr std::array<std::uint32_t, 1u> count_value{kMaximum};
  constexpr std::array<std::uint32_t, 1u> initial{0u};
  std::array<std::uint64_t, 2u> output_hashes{};
  Fingerprint pipeline_fingerprint{};

  for (std::size_t scenario = 0u; scenario < values.size(); ++scenario) {
    std::array<std::uint32_t, kMaximum> initial_window{};
    initial_window.fill(kSentinel);
    auto count = device.upload<std::uint32_t>(count_value);
    auto outer = device.upload<std::uint32_t>(initial);
    auto input_values = device.upload<std::uint32_t>(values[scenario]);
    auto input_targets = device.upload<std::uint32_t>(targets[scenario]);
    auto final = device.upload<std::uint32_t>(initial);
    auto window_target = device.upload<std::uint32_t>(initial_window);
    if (!count || !outer || !input_values || !input_targets || !final ||
        !window_target) {
      return 1;
    }

    const auto body = tile_repeat<0u>(seed, fold);
    auto builder = pipeline(device);
    builder.windows<kMaximum, kTile>(
        body, rund::compute::window(*count),
        read(*outer, *input_values, *input_targets), write_final(*final),
        write_window(*window_target));
    const auto plan = builder.plan();
    if (!plan || plan->outer_window_count != kOuter ||
        plan->prepared_template_count != kTemplates ||
        plan->prepared_command_count != kCommands ||
        plan->publish_count != kOuter + 1u ||
        plan->publish_bytes != (kMaximum + 1u) * sizeof(std::uint32_t)) {
      return 2;
    }
    auto prepared = std::move(builder).prepare();
    if (!prepared || !prepared->run()) {
      return 3;
    }
    std::array<std::uint32_t, 1u> first_final{};
    std::array<std::uint32_t, kMaximum> first_window{};
    if (!prepared->read(*final, first_final) ||
        !prepared->read(*window_target, first_window)) {
      return 4;
    }
    const MemoryStats before_warm = prepared->memory();
    if (!prepared->run()) {
      return 5;
    }
    const Stats stats = prepared->stats();
    const MemoryStats after_warm = prepared->memory();
    std::array<std::uint32_t, 1u> actual_final{};
    std::array<std::uint32_t, kMaximum> actual_window{};
    std::array<std::uint32_t, kMaximum> expected_window{};
    expected_window[kMaximum - 1u] = expected[scenario];
    if (!prepared->read(*final, actual_final) ||
        !prepared->read(*window_target, actual_window) ||
        prepared->generation() != 2u ||
        actual_final[0u] != expected[scenario] ||
        actual_window != expected_window || first_final != actual_final ||
        first_window != actual_window ||
        !NoAllocation(before_warm, after_warm) || !WarmSetupClean(stats) ||
        stats.command_submits != (backend == Backend::Cpu ? 0u : 1u) ||
        stats.pipeline.executed_outer_window_count != kOuter ||
        stats.pipeline.skipped_outer_window_count != 0u ||
        stats.publication.generation != 2u ||
        stats.publication.discard_count != 0u) {
      std::fprintf(
          stderr,
          "window output scatter backend=%u scenario=%llu "
          "final=%u/%u high=%u generation=%llu outer=%llu "
          "discard=%llu\n",
          static_cast<unsigned>(backend),
          static_cast<unsigned long long>(scenario), actual_final[0u],
          expected[scenario], actual_window[kMaximum - 1u],
          static_cast<unsigned long long>(prepared->generation()),
          static_cast<unsigned long long>(
              stats.pipeline.executed_outer_window_count),
          static_cast<unsigned long long>(stats.publication.discard_count));
      return 6;
    }
    std::array<std::uint32_t, kMaximum + 1u> raw{};
    raw[0u] = actual_final[0u];
    std::copy(actual_window.begin(), actual_window.end(), raw.begin() + 1u);
    output_hashes[scenario] = Hash(raw.data(), sizeof(raw));
    if (scenario == 0u) {
      pipeline_fingerprint = prepared->fingerprint();
    } else if (pipeline_fingerprint != prepared->fingerprint()) {
      return 7;
    }
  }

  const WindowOutputIdentity current{
      .scatter_seed = seed.fingerprint(),
      .scatter_fold = fold.fingerprint(),
      .scatter_pipeline = pipeline_fingerprint,
      .scatter_output = Hash(output_hashes.data(), sizeof(output_hashes)),
  };
  if (!current.scatter_seed || !current.scatter_fold ||
      !current.scatter_pipeline || current.scatter_output == 0u) {
    return 8;
  }
  if (identity.scatter_seed) {
    return identity.scatter_seed == current.scatter_seed &&
                   identity.scatter_fold == current.scatter_fold &&
                   identity.scatter_pipeline == current.scatter_pipeline &&
                   identity.scatter_output == current.scatter_output
               ? 0
               : 9;
  }
  identity.scatter_seed = current.scatter_seed;
  identity.scatter_fold = current.scatter_fold;
  identity.scatter_pipeline = current.scatter_pipeline;
  identity.scatter_output = current.scatter_output;
  return 0;
}

template <class Body>
[[nodiscard]] int
CheckLateFailure(Device &device, const Backend backend, const Body &body,
                 const std::uint32_t count_value, const Reason expected_reason,
                 const rund::compute::PipelineNestedPhase phase,
                 const std::uint64_t inner) {
  using namespace rund::compute;
  constexpr std::array<std::uint32_t, 1u> initial{kInitial};
  constexpr std::array<std::uint32_t, 1u> witness_values{0u};
  const std::array<std::uint32_t, 1u> count_values{count_value};
  constexpr std::array<std::uint32_t, 1u> final_values{kSentinel};
  std::array<std::uint32_t, kMaximum> window_values{};
  window_values.fill(kSentinel);
  auto outer = device.upload<std::uint32_t>(initial);
  auto values = device.upload<std::uint32_t>(kValues);
  auto lanes = device.upload<std::uint32_t>(kLanes);
  auto witness = device.upload<std::uint32_t>(witness_values);
  auto count = device.upload<std::uint32_t>(count_values);
  auto final = device.upload<std::uint32_t>(final_values);
  auto appended = device.upload<std::uint32_t>(window_values);
  auto final_observer =
      on(device)
          .map<std::uint32_t>("window-output-final-observe", 1u,
                              [](auto value) { return value; })
          .compile();
  auto final_scratch = device.buffer<std::uint32_t>(1u);
  if (!outer || !values || !lanes || !witness || !count || !final ||
      !appended || !final_observer || !final_scratch) {
    return 1;
  }
  auto builder = pipeline(device);
  builder.windows<kMaximum, kTile>(body, rund::compute::window(*count),
                                   read(*outer, *values, *lanes, *witness),
                                   write_final(*final),
                                   write_window(*appended));
  auto prepared = std::move(builder).prepare();
  const Status failed =
      prepared ? prepared->run() : Status::fail(prepared.reason());
  const Stats stats = prepared ? prepared->stats() : Stats{};
  std::array<std::uint32_t, kMaximum> unread{};
  const Status poisoned_read = prepared ? prepared->read(*appended, unread)
                                        : Status::fail(prepared.reason());
  auto observed = final_observer->run(*final, *final_scratch);
  std::array<std::uint32_t, 1u> actual_final{};
  const bool final_unchanged =
      observed &&
      observed->read(*final_scratch, std::span<std::uint32_t>{actual_final}) &&
      actual_final == final_values;
  const Status rerun =
      prepared ? prepared->run() : Status::fail(prepared.reason());
  if (!prepared || failed || failed.reason() != expected_reason ||
      prepared->generation() != 0u || !prepared->poisoned() || poisoned_read ||
      poisoned_read.reason() != Reason::BufferPoisoned || !final_unchanged ||
      rerun || rerun.reason() != Reason::PipelinePoisoned ||
      stats.pipeline.failed_outer_window != 1u ||
      stats.pipeline.failed_inner_iteration != inner ||
      stats.pipeline.failed_nested_phase != phase ||
      stats.pipeline.executed_outer_window_count != 1u ||
      stats.publication.discard_count != 1u ||
      stats.command_submits != (backend == Backend::Cpu ? 0u : 1u)) {
    std::fprintf(
        stderr,
        "window output failure backend=%u phase=%u status=%u/%u/%u "
        "generation=%llu poison=%u read=%u/%u rerun=%u/%u final=%u/%u "
        "outer=%llu coord=%llu/%llu/%u discard=%llu submit=%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(phase),
        static_cast<unsigned>(failed.ok()),
        static_cast<unsigned>(failed.reason()),
        static_cast<unsigned>(expected_reason),
        static_cast<unsigned long long>(prepared ? prepared->generation() : 0u),
        static_cast<unsigned>(prepared ? prepared->poisoned() : false),
        static_cast<unsigned>(poisoned_read.ok()),
        static_cast<unsigned>(poisoned_read.reason()),
        static_cast<unsigned>(rerun.ok()),
        static_cast<unsigned>(rerun.reason()), actual_final[0u], kSentinel,
        static_cast<unsigned long long>(
            stats.pipeline.executed_outer_window_count),
        static_cast<unsigned long long>(stats.pipeline.failed_outer_window),
        static_cast<unsigned long long>(stats.pipeline.failed_inner_iteration),
        static_cast<unsigned>(stats.pipeline.failed_nested_phase),
        static_cast<unsigned long long>(stats.publication.discard_count),
        static_cast<unsigned long long>(stats.command_submits));
    return 2;
  }
  return 0;
}

template <class Seed, class Fold, class FoldTwo>
[[nodiscard]] int CheckAliases(Device &device, const Seed &seed,
                               const Fold &fold, const FoldTwo &fold_two) {
  using namespace rund::compute;
  constexpr std::array<std::uint32_t, 1u> scalar{1u};
  auto outer = device.upload<std::uint32_t>(scalar);
  auto values = device.upload<std::uint32_t>(kValues);
  auto lanes = device.upload<std::uint32_t>(kLanes);
  auto witness = device.upload<std::uint32_t>(scalar);
  auto count = device.upload<std::uint32_t>(scalar);
  auto final = device.upload<std::uint32_t>(scalar);
  auto shared = device.buffer<std::uint32_t>(2u * kMaximum);
  if (!outer || !values || !lanes || !witness || !count || !final || !shared) {
    return 1;
  }
  auto first = shared->view(0u, kMaximum);
  auto second = shared->view(kMaximum, kMaximum);
  if (!first || !second) {
    return 2;
  }
  const auto two = tile_repeat<0u>(seed, fold_two);
  auto duplicate = pipeline(device);
  duplicate.windows<kMaximum, kTile>(two, rund::compute::window(*count),
                                     read(*outer, *values, *lanes, *witness),
                                     write_final(*final),
                                     write_window(*first, *second));
  const auto duplicate_plan = duplicate.plan();
  if (duplicate_plan || duplicate_plan.reason() != Reason::BindingDuplicate) {
    std::fprintf(stderr, "window output duplicate reason=%u\n",
                 static_cast<unsigned>(duplicate_plan.reason()));
    return 3;
  }

  std::array<std::uint32_t, kMaximum> aliased_values{};
  aliased_values[0u] = 1u;
  auto aliased = device.upload<std::uint32_t>(aliased_values);
  if (!aliased) {
    return 4;
  }
  auto count_view = aliased->view(0u, 1u);
  if (!count_view) {
    return 5;
  }
  const auto one = tile_repeat<0u>(seed, fold);
  auto overlap = pipeline(device);
  overlap.windows<kMaximum, kTile>(one, rund::compute::window(*count_view),
                                   read(*outer, *values, *lanes, *witness),
                                   write_final(*final), write_window(*aliased));
  const auto overlap_plan = overlap.plan();
  if (overlap_plan ||
      overlap_plan.reason() != Reason::BindingAliasUnsupported) {
    std::fprintf(stderr, "window output overlap reason=%u\n",
                 static_cast<unsigned>(overlap_plan.reason()));
  }
  return !overlap_plan &&
                 overlap_plan.reason() == Reason::BindingAliasUnsupported
             ? 0
             : 6;
}

} // namespace

[[nodiscard]] int CheckWindowOutput(Device &device, const Backend backend,
                                    WindowOutputIdentity &identity) {
  auto seed = SeedProgram<false>(device);
  auto seed_fault = SeedProgram<true>(device);
  auto action = ActionProgram<false>(device);
  auto action_fault = ActionProgram<true>(device);
  auto fold = FoldProgram(device);
  auto high_index_fold = FoldProgram<false, true>(device);
  auto fold_two = FoldProgram<true>(device);
  auto priority_fold = PriorityFoldProgram(device);
  auto scatter_seed = ScatterSeedProgram(device);
  auto scatter_fold = ScatterFoldProgram(device);
  auto downstream = DownstreamProgram(device);
  auto count_advance = CountAdvanceProgram(device);
  auto zero_seed = ZeroPrefixSeedProgram(device);
  auto zero_fold = ZeroPrefixFoldProgram(device);
  if (!seed || !seed_fault || !action || !action_fault || !fold ||
      !high_index_fold || !fold_two || !priority_fold || !scatter_seed ||
      !scatter_fold || !downstream || !count_advance || !zero_seed ||
      !zero_fold) {
    return 1;
  }
  if (!PublicationFingerprintV3Golden()) {
    return 2;
  }
  if (!PublicationSourceCoordinates()) {
    return 3;
  }
  if (const int aliases = CheckOrdinaryAliasAuthority(device); aliases != 0) {
    return 140 + aliases;
  }
  if (const int subview = CheckAliasSubviewRouting(device); subview != 0) {
    return 150 + subview;
  }
  if (const int mutation = CheckSealedPublicationMutation(device, *seed, *fold);
      mutation != 0) {
    return 3 + mutation;
  }
  if (const int mutation =
          CheckPublicationJobBindingMutation(device, *seed, *fold);
      mutation != 0) {
    return 40 + mutation;
  }
  if (const int parity =
          CheckTransactionalCountParity(device, *seed, *fold, *count_advance);
      parity != 0) {
    return 120 + parity;
  }
  if (const int target =
          CheckStatePairPublicationTargetRejected(device, *seed, *fold);
      target != 0) {
    return 130 + target;
  }
  for (const std::uint32_t count : kCounts) {
    const int checked = CheckCount(device, backend, *seed, *fold, count,
                                   kValues, false, identity);
    if (checked != 0) {
      return 10 + checked;
    }
  }
  if (const int high_index =
          CheckCount(device, backend, *seed, *high_index_fold, kMaximum,
                     kValues, true, identity);
      high_index != 0) {
    return 20 + high_index;
  }
  if (const int scatter = CheckScatterConflicts(device, backend, *scatter_seed,
                                                *scatter_fold, identity);
      scatter != 0) {
    return 30 + scatter;
  }
  if (const int downstream_read = CheckDownstreamRead(
          device, backend, *seed, *fold, *downstream, identity);
      downstream_read != 0) {
    return 40 + downstream_read;
  }
  if (const int zero_prefix = CheckZeroRecurrentPrefix(
          device, backend, *zero_seed, *zero_fold, identity);
      zero_prefix != 0) {
    return 60 + zero_prefix;
  }
  if (const int arity = CheckPublicationArity(device, *seed, *action);
      arity != 0) {
    return 70 + arity;
  }
  const auto seed_failure = rund::compute::tile_repeat<0u>(*seed_fault, *fold);
  if (const int failure = CheckLateFailure(
          device, backend, seed_failure, 5u, Reason::GatherIndexOutOfRange,
          rund::compute::PipelineNestedPhase::Seed,
          rund::compute::PipelineStats::no_coordinate);
      failure != 0) {
    return 80 + failure;
  }
  const auto action_failure =
      rund::compute::tile_repeat<1u>(*seed, *action_fault, *fold);
  if (const int failure = CheckLateFailure(
          device, backend, action_failure, 5u, Reason::GatherIndexOutOfRange,
          rund::compute::PipelineNestedPhase::Action, 0u);
      failure != 0) {
    return 90 + failure;
  }
  const auto priority_failure =
      rund::compute::tile_repeat<0u>(*seed, *priority_fold);
  if (const int failure = CheckLateFailure(
          device, backend, priority_failure, 8u, Reason::ScatterIndexOutOfRange,
          rund::compute::PipelineNestedPhase::Fold,
          rund::compute::PipelineStats::no_coordinate);
      failure != 0) {
    return 100 + failure;
  }
  const int aliases = CheckAliases(device, *seed, *fold, *fold_two);
  return aliases == 0 ? 0 : 110 + aliases;
}

} // namespace rund::node::test_contract::window
