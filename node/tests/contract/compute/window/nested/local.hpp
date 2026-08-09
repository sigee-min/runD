#pragma once

#include <rund/compute.hpp>
#include <rund/compute/pipeline.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

namespace rund::node::test_contract::window {

inline constexpr std::size_t kMaximum = 10u;
inline constexpr std::size_t kTile = 4u;
inline constexpr std::size_t kInner = 3u;
inline constexpr std::size_t kDomain = 64u;
inline constexpr std::size_t kOuter = CeilDiv(kMaximum, kTile);
inline constexpr std::size_t kTemplates = kOuter + kInner + 3u;
inline constexpr std::size_t kPreparedTemplates =
    kOuter + (kInner == 1u ? 1u : 2u) + 3u;
inline constexpr std::size_t kCommands = kOuter * (kInner + 2u);
inline constexpr std::uint32_t kOuterSeed = 7u;
inline constexpr std::uint32_t kSentinel = 0xA5A55A5Au;
inline constexpr std::array<std::uint32_t, kMaximum> kQueue{
    63u, 0u, 47u, 47u, 1u, 62u, 31u, 5u, 60u, 2u};
inline constexpr auto kDomainValues = [] {
  std::array<std::uint32_t, kDomain> values{};
  for (std::size_t index = 0u; index < values.size(); ++index) {
    values[index] = static_cast<std::uint32_t>(3u * index + 1u);
  }
  return values;
}();
inline constexpr std::array<std::uint32_t, 7u> kCounts{
    0u,
    1u,
    static_cast<std::uint32_t>(kTile - 1u),
    static_cast<std::uint32_t>(kTile),
    static_cast<std::uint32_t>(kTile + 1u),
    static_cast<std::uint32_t>(kMaximum),
    static_cast<std::uint32_t>(kMaximum + 1u)};

using NestedSeed = rund::compute::Program<rund::compute::Outputs<
    rund::compute::Scalar<std::uint32_t>, rund::compute::Scalar<std::uint32_t>>(
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t)>;
using NestedAction =
    rund::compute::Program<std::uint32_t(std::uint32_t, std::uint32_t)>;
using NestedFold = rund::compute::Program<std::uint32_t(
    std::uint32_t, std::uint32_t, std::uint32_t)>;
using NestedTerminalSeed =
    rund::compute::Program<std::uint32_t(std::uint32_t, std::uint32_t)>;
using NestedTerminalFold =
    rund::compute::Program<rund::compute::Outputs<std::uint32_t, std::uint32_t>(
        std::uint32_t, std::uint32_t, std::uint32_t)>;
using NestedFailureAction =
    rund::compute::Program<std::uint32_t(std::uint32_t)>;
using NestedFailureFold =
    rund::compute::Program<std::uint32_t(std::uint32_t, std::uint32_t)>;
using NestedSeedResult = rund::compute::Result<NestedSeed>;
using NestedActionResult = rund::compute::Result<NestedAction>;
using NestedFoldResult = rund::compute::Result<NestedFold>;

[[nodiscard]] NestedSeedResult MakeNestedSeedProgram(rund::compute::Device &);
[[nodiscard]] NestedActionResult
MakeNestedActionProgram(rund::compute::Device &);
[[nodiscard]] NestedActionResult
MakeNestedMemoryActionProgram(rund::compute::Device &);
[[nodiscard]] NestedFoldResult MakeNestedFoldProgram(rund::compute::Device &);
[[nodiscard]] NestedSeedResult MakeNestedSeed4Program(rund::compute::Device &);
[[nodiscard]] NestedSeedResult MakeNestedSeed8Program(rund::compute::Device &);
[[nodiscard]] NestedSeedResult
MakeNestedSeed1024Program(rund::compute::Device &);
[[nodiscard]] NestedSeedResult
MakeNestedSeed2048Program(rund::compute::Device &);
[[nodiscard]] NestedSeedResult
MakeNestedSeed3072Program(rund::compute::Device &);
[[nodiscard]] NestedSeedResult
MakeNestedSeed516096Program(rund::compute::Device &);
[[nodiscard]] NestedSeedResult
MakeNestedProductSeedProgram(rund::compute::Device &);
[[nodiscard]] rund::compute::Result<NestedTerminalSeed>
MakeNestedTerminalSeedProgram(rund::compute::Device &);
[[nodiscard]] rund::compute::Result<NestedTerminalFold>
MakeNestedTerminalFoldProgram(rund::compute::Device &);
[[nodiscard]] rund::compute::Result<NestedTerminalSeed>
MakeNestedFailureSeedProgram(rund::compute::Device &, bool);
[[nodiscard]] rund::compute::Result<NestedFailureAction>
MakeNestedFailureActionProgram(rund::compute::Device &, std::uint32_t);
[[nodiscard]] rund::compute::Result<NestedFailureFold>
MakeNestedFailureFoldProgram(rund::compute::Device &, bool);
[[nodiscard]] std::uint32_t SerialOracle(std::uint32_t) noexcept;
[[nodiscard]] std::size_t ActionOwnerCount(const rund::compute::Pipeline &,
                                           rund::compute::graph::Fingerprint);

[[nodiscard]] bool CpuRouteOwnershipIsExact(const rund::compute::Pipeline &);
[[nodiscard]] bool WarmSetupClean(const rund::compute::Stats &) noexcept;
[[nodiscard]] bool PreparedShape(const rund::compute::Pipeline &);
[[nodiscard]] bool NestedPlanShape(const rund::compute::PipelinePlan &,
                                   const NestedSeed &, const NestedAction &,
                                   const NestedFold &);
[[nodiscard]] bool RuntimeShape(const rund::compute::Stats &,
                                rund::compute::Backend, std::uint32_t) noexcept;

[[nodiscard]] int CheckNestedCount(rund::compute::Device &,
                                   rund::compute::Backend, const NestedSeed &,
                                   const NestedAction &, const NestedFold &,
                                   std::uint32_t);
[[nodiscard]] int CheckTransactionalBindingIdentity(rund::compute::Device &,
                                                    rund::compute::Backend);
[[nodiscard]] int CheckNestedTerminal(rund::compute::Device &,
                                      rund::compute::Backend);
[[nodiscard]] int CheckNestedFailures(rund::compute::Device &,
                                      rund::compute::Backend);
[[nodiscard]] int CheckNestedProfileCase(rund::compute::Device &,
                                         rund::compute::Backend,
                                         const NestedSeed &,
                                         const NestedAction &,
                                         const NestedFold &);
[[nodiscard]] int CheckNestedRetainedReuse(rund::compute::Device &,
                                           const NestedSeed &,
                                           const NestedFold &);
[[nodiscard]] int CheckNestedComposition(rund::compute::Device &,
                                         rund::compute::Backend,
                                         const NestedSeed &,
                                         const NestedAction &,
                                         const NestedFold &);
[[nodiscard]] int CheckNestedAggregateStats(rund::compute::Device &,
                                            rund::compute::Backend);
[[nodiscard]] int CheckDormantAggregateRoutes(rund::compute::Device &,
                                              rund::compute::Backend);
[[nodiscard]] int CheckWorkspaceObservationCapacity(rund::compute::Device &);
[[nodiscard]] int CheckNestedAggregateSeedFailures(rund::compute::Device &,
                                                   rund::compute::Backend,
                                                   const NestedSeed &,
                                                   const NestedAction &,
                                                   const NestedFold &);
[[nodiscard]] int CheckNestedMaximumPlan(rund::compute::Device &,
                                         const NestedAction &,
                                         const NestedFold &);
[[nodiscard]] int CheckProductPlan(rund::compute::Device &,
                                   rund::compute::Backend);
[[nodiscard]] int CheckNestedWindow(rund::compute::Device &,
                                    rund::compute::Backend);

} // namespace rund::node::test_contract::window
