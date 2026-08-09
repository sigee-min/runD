#pragma once

#include "../local.hpp"

#include <rund/compute.hpp>
#include <rund/compute/pipeline.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

namespace rund::node::test_contract::window {

inline constexpr std::size_t kOutputMaximum = 16u;
inline constexpr std::size_t kOutputTile = 4u;
inline constexpr std::size_t kOutputOuter = kOutputMaximum / kOutputTile;
inline constexpr std::size_t kOutputTemplates = kOutputOuter + 3u;
inline constexpr std::size_t kOutputCommands = 2u * kOutputOuter;
inline constexpr std::size_t kOutputWindowOffset = 2u;
inline constexpr std::size_t kOutputWindowBacking = kOutputMaximum + 4u;
inline constexpr std::uint32_t kOutputInitial = 7u;
inline constexpr std::uint32_t kOutputSentinel = 0xA5A55A5Au;
inline constexpr std::uint32_t kOutputWindowBias = 1000u;
inline constexpr std::uint32_t kOutputHighIndexValue = 0xF0000001u;
inline constexpr std::array<std::uint32_t, kOutputMaximum> kOutputValues{
    11u, 22u,  22u,  44u,  55u,  66u,  77u,  88u,
    99u, 110u, 121u, 132u, 143u, 154u, 165u, 0xF0000001u};
inline constexpr std::array<std::uint32_t, kOutputTile> kOutputLanes{0u, 1u, 2u,
                                                                     3u};
inline constexpr std::array<std::uint32_t, 6u> kOutputCounts{0u, 1u, 3u,
                                                             4u, 5u, 16u};

using OutputSeed = rund::compute::Program<rund::compute::Outputs<
    std::uint32_t, rund::compute::Scalar<std::uint32_t>,
    rund::compute::Scalar<std::uint32_t>, std::uint32_t>(
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t)>;
using OutputAction =
    rund::compute::Program<rund::compute::Outputs<std::uint32_t, std::uint32_t,
                                                  std::uint32_t, std::uint32_t>(
        std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t)>;
using OutputFold =
    rund::compute::Program<rund::compute::Outputs<std::uint32_t, std::uint32_t>(
        std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
        std::uint32_t)>;
using OutputFoldTwo = rund::compute::Program<rund::compute::Outputs<
    std::uint32_t, std::uint32_t, std::uint32_t>(
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t)>;
using OutputScatterSeed =
    rund::compute::Program<rund::compute::Outputs<std::uint32_t, std::uint32_t>(
        std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t)>;
using OutputScatterFold =
    rund::compute::Program<rund::compute::Outputs<std::uint32_t, std::uint32_t>(
        std::uint32_t, std::uint32_t, std::uint32_t)>;
using OutputUnary = rund::compute::Program<std::uint32_t(std::uint32_t)>;
using OutputZero =
    rund::compute::Program<rund::compute::Outputs<std::uint32_t, std::uint32_t>(
        std::uint32_t, std::uint32_t, std::uint32_t)>;

[[nodiscard]] rund::compute::Result<OutputSeed>
MakeOutputSeedProgram(rund::compute::Device &, bool);
[[nodiscard]] rund::compute::Result<OutputAction>
MakeOutputActionProgram(rund::compute::Device &, bool);
[[nodiscard]] rund::compute::Result<OutputFold>
MakeOutputFoldProgram(rund::compute::Device &);
[[nodiscard]] rund::compute::Result<OutputFold>
MakeOutputHighIndexFoldProgram(rund::compute::Device &);
[[nodiscard]] rund::compute::Result<OutputFoldTwo>
MakeOutputFoldTwoProgram(rund::compute::Device &);
[[nodiscard]] rund::compute::Result<OutputFold>
MakeOutputPriorityFoldProgram(rund::compute::Device &);
[[nodiscard]] rund::compute::Result<OutputScatterSeed>
MakeOutputScatterSeedProgram(rund::compute::Device &);
[[nodiscard]] rund::compute::Result<OutputScatterFold>
MakeOutputScatterFoldProgram(rund::compute::Device &);
[[nodiscard]] rund::compute::Result<OutputUnary>
MakeOutputDownstreamProgram(rund::compute::Device &);
[[nodiscard]] rund::compute::Result<OutputUnary>
MakeOutputCountAdvanceProgram(rund::compute::Device &);
[[nodiscard]] rund::compute::Result<OutputZero>
MakeOutputZeroSeedProgram(rund::compute::Device &);
[[nodiscard]] rund::compute::Result<OutputZero>
MakeOutputZeroFoldProgram(rund::compute::Device &);

[[nodiscard]] std::uint32_t
ExpectedFinal(const std::array<std::uint32_t, kOutputMaximum> &,
              std::uint32_t) noexcept;
[[nodiscard]] std::array<std::uint32_t, kOutputMaximum>
ExpectedWindow(const std::array<std::uint32_t, kOutputMaximum> &,
               std::uint32_t) noexcept;
[[nodiscard]] std::array<std::uint32_t, kOutputMaximum>
ExpectedHighIndexWindow() noexcept;
[[nodiscard]] bool OutputWarmSetupClean(const rund::compute::Stats &) noexcept;
[[nodiscard]] bool PublicationFingerprintV3Golden();
[[nodiscard]] bool PublicationSourceCoordinates();

[[nodiscard]] int CheckOutputPublicationArity(rund::compute::Device &,
                                              const OutputSeed &,
                                              const OutputAction &);
[[nodiscard]] int CheckOutputBuildAllocationRollback(rund::compute::Device &,
                                                     const OutputSeed &,
                                                     const OutputFold &);
[[nodiscard]] int CheckOrdinaryAliasAuthority(rund::compute::Device &);
[[nodiscard]] int CheckAliasSubviewRouting(rund::compute::Device &);
[[nodiscard]] int CheckOutputSealedPublicationMutation(rund::compute::Device &,
                                                       const OutputSeed &,
                                                       const OutputFold &);
[[nodiscard]] int CheckOutputPublicationJobBindingMutation(
    rund::compute::Device &, const OutputSeed &, const OutputFold &);
[[nodiscard]] int CheckOutputTransactionalCountParity(rund::compute::Device &,
                                                      const OutputSeed &,
                                                      const OutputFold &,
                                                      const OutputUnary &);
[[nodiscard]] int CheckOutputStatePairPublicationTargetRejected(
    rund::compute::Device &, const OutputSeed &, const OutputFold &);
[[nodiscard]] int
CheckOutputCount(rund::compute::Device &, rund::compute::Backend,
                 const OutputSeed &, const OutputFold &, std::uint32_t,
                 const std::array<std::uint32_t, kOutputMaximum> &, bool,
                 WindowOutputIdentity &);
[[nodiscard]] int
CheckOutputDownstreamRead(rund::compute::Device &, rund::compute::Backend,
                          const OutputSeed &, const OutputFold &,
                          const OutputUnary &, WindowOutputIdentity &);
[[nodiscard]] int CheckOutputZeroRecurrentPrefix(rund::compute::Device &,
                                                 rund::compute::Backend,
                                                 const OutputZero &,
                                                 const OutputZero &,
                                                 WindowOutputIdentity &);
[[nodiscard]] int CheckOutputScatterConflicts(rund::compute::Device &,
                                              rund::compute::Backend,
                                              const OutputScatterSeed &,
                                              const OutputScatterFold &,
                                              WindowOutputIdentity &);
[[nodiscard]] int
CheckOutputLateFailures(rund::compute::Device &, rund::compute::Backend,
                        const OutputSeed &, const OutputSeed &,
                        const OutputAction &, const OutputFold &,
                        const OutputFold &);
[[nodiscard]] int CheckOutputAliases(rund::compute::Device &,
                                     const OutputSeed &, const OutputFold &,
                                     const OutputFoldTwo &);

} // namespace rund::node::test_contract::window
