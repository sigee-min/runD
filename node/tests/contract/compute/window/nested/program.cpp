#include "../../pipeline/local.hpp"
#include "../local.hpp"
#include "local.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/cpu/prepared.hpp"
#include "src/compute/cpu/run/state.hpp"
#include "src/compute/memory/cpu.hpp"
#include "src/compute/pipeline/state.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace rund::node::test_contract::window {
template <std::size_t Maximum, std::size_t Tile>
[[nodiscard]] auto SeedProgram(rund::compute::Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(Maximum)
      .zip_input<std::uint32_t>(kDomain)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto queue, auto domain, auto total, auto ordinal) {
        auto current = resident<Maximum, Tile>(total, ordinal);
        auto active_ordinals = queue.gather(current.items());
        auto sum = domain.gather(active_ordinals).reduce(Reduce::Sum);
        return outputs(sum, current.count());
      })
      .compile();
}
[[nodiscard]] auto ActionProgram(rund::compute::Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto value, auto tile_count) {
        return value.combine(
            "nested-window-action", tile_count,
            [](auto current, auto count) { return current + count; });
      })
      .compile();
}

[[nodiscard]] auto MemoryActionProgram(rund::compute::Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto value, auto tile_count) {
        auto prefix = value.scan(Scan::InclusiveSum);
        return prefix.combine(
            "nested-window-memory-action", tile_count,
            [](auto current, auto count) { return current + count; });
      })
      .compile();
}

[[nodiscard]] auto FoldProgram(rund::compute::Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto outer, auto tile, auto tile_count) {
        auto checked = tile.combine(
            "nested-window-fold-tail", tile_count,
            [](auto value, auto count) { return value + count * 0u; });
        return outer.combine(
            "nested-window-fold", checked,
            [](auto left, auto right) { return left + right; });
      })
      .compile();
}

[[nodiscard]] auto TerminalSeedProgram(rund::compute::Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto total, auto ordinal) {
        (void)total;
        return ordinal.map("nested-window-terminal-seed",
                           [](auto value) { return value + 1u; });
      })
      .compile();
}

[[nodiscard]] auto TerminalFoldProgram(rund::compute::Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto outer, auto terminal, auto tile) {
        auto next =
            outer.combine("nested-window-terminal-fold", tile,
                          [](auto left, auto right) { return left + right; });
        auto stopped = terminal.map("nested-window-terminal-value",
                                    [](auto value) { return value * 0u + 7u; });
        return outputs(next, stopped);
      })
      .compile();
}

template <bool Fault>
[[nodiscard]] auto FailureSeedProgram(rund::compute::Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto total, auto ordinal) {
        (void)total;
        if constexpr (Fault) {
          auto fault =
              ordinal.map("nested-window-seed-fault-index", [](auto value) {
                return select(value == 1u, 1u, 0u);
              });
          auto checked = ordinal.gather(fault).scalar();
          return checked.map("nested-window-seed-fault-value",
                             [](auto value) { return value * 16u; });
        } else {
          return ordinal.map("nested-window-failure-seed",
                             [](auto value) { return value * 16u; });
        }
      })
      .compile();
}

template <std::uint32_t Target>
[[nodiscard]] auto FailureActionProgram(rund::compute::Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .branch([](auto value) {
        auto fault =
            value.map("nested-window-action-fault-index", [](auto current) {
              return select(current == Target, 1u, 0u);
            });
        auto checked = value.gather(fault).scalar();
        return checked.map("nested-window-action-fault-value",
                           [](auto current) { return current + 1u; });
      })
      .compile();
}

template <bool Fault>
[[nodiscard]] auto FailureFoldProgram(rund::compute::Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto outer, auto tile) {
        if constexpr (Fault) {
          auto fault =
              tile.map("nested-window-fold-fault-index",
                       [](auto value) { return select(value == 19u, 1u, 0u); });
          auto checked = tile.gather(fault).scalar();
          return outer.combine(
              "nested-window-fold-fault", checked,
              [](auto left, auto right) { return left + right; });
        } else {
          return outer.combine(
              "nested-window-failure-fold", tile,
              [](auto left, auto right) { return left + right; });
        }
      })
      .compile();
}

NestedSeedResult MakeNestedSeedProgram(rund::compute::Device &device) {
  return SeedProgram<kMaximum, kTile>(device);
}
NestedActionResult MakeNestedActionProgram(rund::compute::Device &device) {
  return ActionProgram(device);
}
NestedActionResult
MakeNestedMemoryActionProgram(rund::compute::Device &device) {
  return MemoryActionProgram(device);
}
NestedFoldResult MakeNestedFoldProgram(rund::compute::Device &device) {
  return FoldProgram(device);
}
NestedSeedResult MakeNestedSeed4Program(rund::compute::Device &device) {
  return SeedProgram<4u, 4u>(device);
}
NestedSeedResult MakeNestedSeed8Program(rund::compute::Device &device) {
  return SeedProgram<8u, 4u>(device);
}
NestedSeedResult MakeNestedSeed1024Program(rund::compute::Device &device) {
  return SeedProgram<1024u, 1024u>(device);
}
NestedSeedResult MakeNestedSeed2048Program(rund::compute::Device &device) {
  return SeedProgram<2048u, 1024u>(device);
}
NestedSeedResult MakeNestedSeed3072Program(rund::compute::Device &device) {
  return SeedProgram<3072u, 1024u>(device);
}
NestedSeedResult MakeNestedSeed516096Program(rund::compute::Device &device) {
  return SeedProgram<516096u, 1024u>(device);
}
NestedSeedResult MakeNestedProductSeedProgram(rund::compute::Device &device) {
  return SeedProgram<33u, 1u>(device);
}
rund::compute::Result<NestedTerminalSeed>
MakeNestedTerminalSeedProgram(rund::compute::Device &device) {
  return TerminalSeedProgram(device);
}
rund::compute::Result<NestedTerminalFold>
MakeNestedTerminalFoldProgram(rund::compute::Device &device) {
  return TerminalFoldProgram(device);
}
rund::compute::Result<NestedTerminalSeed>
MakeNestedFailureSeedProgram(rund::compute::Device &device, const bool fault) {
  return fault ? FailureSeedProgram<true>(device)
               : FailureSeedProgram<false>(device);
}
rund::compute::Result<NestedFailureAction>
MakeNestedFailureActionProgram(rund::compute::Device &device,
                               const std::uint32_t target) {
  switch (target) {
  case 16u:
    return FailureActionProgram<16u>(device);
  case 17u:
    return FailureActionProgram<17u>(device);
  default:
    return FailureActionProgram<18u>(device);
  }
}
rund::compute::Result<NestedFailureFold>
MakeNestedFailureFoldProgram(rund::compute::Device &device, const bool fault) {
  return fault ? FailureFoldProgram<true>(device)
               : FailureFoldProgram<false>(device);
}

} // namespace rund::node::test_contract::window
