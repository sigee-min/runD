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
static_assert(kOutputOuter == 4u);
static_assert(kOutputTemplates == 7u);
static_assert(kOutputCommands == 8u);

template <bool Fault> [[nodiscard]] auto SeedProgram(Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(kOutputMaximum)
      .zip_input<std::uint32_t>(kOutputTile)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto values, auto lanes, auto witness, auto total,
                 auto ordinal) {
        auto current = resident<kOutputMaximum, kOutputTile>(total, ordinal);
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
      .input<std::uint32_t>(kOutputTile)
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
      .zip_input<std::uint32_t>(kOutputTile)
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
                  return select(lane == kOutputTile - 1u,
                                select(current == kOutputOuter - 1u,
                                       kOutputHighIndexValue, kOutputSentinel),
                                kOutputSentinel);
                });
          } else {
            return tile.map("window-output-first", [](auto value) {
              return value + kOutputWindowBias;
            });
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

[[nodiscard]] auto PriorityFoldProgram(Device &device) {
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
                  select(lane == 1u, static_cast<std::uint32_t>(kOutputTile),
                         select(lane == 2u, 0u, select(lane == 3u, 2u, lane)));
              return select(current == 1u, failed, lane);
            });
        auto sum = tile.reduce(Reduce::Sum);
        auto next =
            outer.combine("window-output-priority-fold", sum,
                          [](auto state, auto value) { return state + value; });
        auto scattered = tile.scatter(targets, {.count = kOutputTile});
        return outputs(next, scattered);
      })
      .compile();
}

[[nodiscard]] auto ScatterSeedProgram(Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(kOutputTile)
      .zip_input<std::uint32_t>(kOutputTile)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto values, auto targets, auto total, auto ordinal) {
        (void)total;
        auto selected = values.combine(
            "window-output-scatter-select", ordinal.scalar(),
            [](auto value, auto outer) {
              return select(outer == kOutputOuter - 1u, value, 0u);
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
      .zip_input<std::uint32_t>(kOutputTile)
      .zip_input<std::uint32_t>(kOutputTile)
      .branch([](auto outer, auto values, auto targets) {
        auto sum = values.reduce(Reduce::Sum);
        auto next =
            outer.combine("window-output-scatter-fold", sum,
                          [](auto state, auto value) { return state + value; });
        auto scattered =
            values.scatter_reduce(targets, kOutputTile, Reduce::Sum);
        return outputs(next, scattered);
      })
      .compile();
}

[[nodiscard]] auto DownstreamProgram(Device &device) {
  using namespace rund::compute;
  return on(device)
      .map<std::uint32_t>("window-output-downstream", kOutputMaximum,
                          [](auto value) { return value; })
      .compile();
}

[[nodiscard]] auto CountAdvanceProgram(Device &device) {
  using namespace rund::compute;
  return on(device)
      .map<std::uint32_t>("window-output-count-advance", 1u,
                          [](auto value) { return value + kOutputTile; })
      .compile();
}

[[nodiscard]] auto ZeroPrefixSeedProgram(Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(kOutputTile)
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
      .zip_input<std::uint32_t>(kOutputTile)
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

[[nodiscard]] std::uint32_t
ExpectedFinal(const std::array<std::uint32_t, kOutputMaximum> &values,
              const std::uint32_t count) noexcept {
  std::uint32_t result = kOutputInitial;
  for (std::size_t index = 0u; index < count; ++index) {
    result += values[index];
  }
  return result;
}

[[nodiscard]] std::array<std::uint32_t, kOutputMaximum>
ExpectedWindow(const std::array<std::uint32_t, kOutputMaximum> &values,
               const std::uint32_t count) noexcept {
  std::array<std::uint32_t, kOutputMaximum> result{};
  result.fill(kOutputSentinel);
  for (std::size_t index = 0u; index < count; ++index) {
    result[index] = values[index] + kOutputWindowBias;
  }
  return result;
}

[[nodiscard]] std::array<std::uint32_t, kOutputMaximum>
ExpectedHighIndexWindow() noexcept {
  std::array<std::uint32_t, kOutputMaximum> result{};
  result.fill(kOutputSentinel);
  result[kOutputMaximum - 1u] = kOutputHighIndexValue;
  return result;
}

rund::compute::Result<OutputSeed> MakeOutputSeedProgram(Device &device,
                                                        const bool fault) {
  return fault ? SeedProgram<true>(device) : SeedProgram<false>(device);
}
rund::compute::Result<OutputAction> MakeOutputActionProgram(Device &device,
                                                            const bool fault) {
  return fault ? ActionProgram<true>(device) : ActionProgram<false>(device);
}
rund::compute::Result<OutputFold> MakeOutputFoldProgram(Device &device) {
  return FoldProgram(device);
}
rund::compute::Result<OutputFold>
MakeOutputHighIndexFoldProgram(Device &device) {
  return FoldProgram<false, true>(device);
}
rund::compute::Result<OutputFoldTwo> MakeOutputFoldTwoProgram(Device &device) {
  return FoldProgram<true>(device);
}
rund::compute::Result<OutputFold>
MakeOutputPriorityFoldProgram(Device &device) {
  return PriorityFoldProgram(device);
}
rund::compute::Result<OutputScatterSeed>
MakeOutputScatterSeedProgram(Device &device) {
  return ScatterSeedProgram(device);
}
rund::compute::Result<OutputScatterFold>
MakeOutputScatterFoldProgram(Device &device) {
  return ScatterFoldProgram(device);
}
rund::compute::Result<OutputUnary> MakeOutputDownstreamProgram(Device &device) {
  return DownstreamProgram(device);
}
rund::compute::Result<OutputUnary>
MakeOutputCountAdvanceProgram(Device &device) {
  return CountAdvanceProgram(device);
}
rund::compute::Result<OutputZero> MakeOutputZeroSeedProgram(Device &device) {
  return ZeroPrefixSeedProgram(device);
}
rund::compute::Result<OutputZero> MakeOutputZeroFoldProgram(Device &device) {
  return ZeroPrefixFoldProgram(device);
}

} // namespace rund::node::test_contract::window
