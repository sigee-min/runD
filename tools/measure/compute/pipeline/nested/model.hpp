#pragma once

#include "../model.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace rund::measure::compute::nested_repeat {

inline constexpr std::size_t Maximum = 516096u;
inline constexpr std::size_t Tile = 1024u;
inline constexpr std::size_t Inner = 64u;
inline constexpr std::size_t Repetitions = 256u;
inline constexpr std::size_t Outer = Maximum / Tile;
inline constexpr std::size_t Templates = Outer + Inner + 3u;
inline constexpr std::size_t Commands = Outer * (Inner + 2u);
inline constexpr std::size_t SerialSubmits = Outer * Inner;
inline constexpr std::size_t Domain = 64u;
inline constexpr std::uint32_t OuterSeed = 7u;

// The admitted aggregate owns one parallel tile-reduction command and one
// deterministic ordered finalize/publication command. A larger value proves
// that the canonical fallback stream ran instead of the aggregate hard cut.
inline constexpr std::uint64_t MetalNestedPhysicalCommands = 2u;

static_assert(Maximum % Tile == 0u);
static_assert(Outer == 504u);
static_assert(SerialSubmits == 32256u);
static_assert(Templates == 571u);
static_assert(Commands == 33264u);
static_assert(Inner % 2u == 0u);
static_assert(MetalNestedPhysicalCommands == 2u);
static_assert(Repetitions <= ::rund::compute::PipelineSealedRepetitionCapacity);

struct Fixture final {
  ::rund::compute::Backend backend;
  std::uint32_t expected;
  ::rund::compute::Device device;
  ::rund::compute::Program<std::uint32_t(std::uint32_t)> observe_program;
  ::rund::compute::Buffer<std::uint32_t> serial_seed;
  ::rund::compute::Buffer<std::uint32_t> serial_first;
  ::rund::compute::Buffer<std::uint32_t> serial_second;
  ::rund::compute::Buffer<std::uint32_t> serial_observed;
  ::rund::compute::Buffer<std::uint32_t> nested_output;
  ::rund::compute::Buffer<std::uint32_t> repeated_output;
  ::rund::compute::Buffer<std::uint32_t> sealed_output;
  std::vector<::rund::compute::Pipeline> serial_steps;
  ::rund::compute::PipelinePlan nested_plan;
  ::rund::compute::Pipeline nested;
  ::rund::compute::PipelinePlan repeated_plan;
  ::rund::compute::Pipeline repeated;
  ::rund::compute::PipelinePlan sealed_plan;
  ::rund::compute::Pipeline sealed;
};

struct Measurements final {
  std::vector<double> serial_wall;
  std::vector<double> serial_pair_nested_wall;
  std::vector<double> latency_pair_nested_wall;
  std::vector<double> latency_pair_sealed_wall;
  std::vector<double> repeated_wall;
  std::vector<double> throughput_pair_sealed_wall;
  WarmCounters serial_warm{};
  WarmCounters nested_warm{};
  WarmCounters repeated_warm{};
  WarmCounters sealed_warm{};
  ExecutionCounters serial_counters{};
  ExecutionCounters nested_counters{};
  ExecutionCounters repeated_counters{};
  ExecutionCounters sealed_counters{};
  ::rund::compute::Stats nested_stats{};
  ::rund::compute::Stats repeated_stats{};
  ::rund::compute::Stats sealed_stats{};
  std::size_t serial_pair_serial_first{};
  std::size_t serial_pair_nested_first{};
  std::size_t latency_pair_nested_first{};
  std::size_t latency_pair_sealed_first{};
  std::size_t throughput_pair_repeated_first{};
  std::size_t throughput_pair_sealed_first{};

  void reserve(std::size_t samples);
};

[[nodiscard]] std::optional<Fixture> Prepare(::rund::compute::Backend backend);
[[nodiscard]] bool SerialStepEvidence(::rund::compute::Backend backend,
                                      const ::rund::compute::Stats &stats);
[[nodiscard]] bool NestedEvidence(::rund::compute::Backend backend,
                                  const ::rund::compute::Stats &stats,
                                  const ::rund::compute::PipelinePlan &plan);
[[nodiscard]] bool RunSerial(Fixture &fixture, Measurements &measurements,
                             bool timed);
[[nodiscard]] bool RunNested(Fixture &fixture, Measurements &measurements,
                             std::vector<double> *timings);
[[nodiscard]] bool RunRepeated(Fixture &fixture, Measurements &measurements,
                               std::vector<double> *timings);
[[nodiscard]] bool RunSealed(Fixture &fixture, Measurements &measurements,
                             std::vector<double> *timings);
[[nodiscard]] bool Sample(Fixture &fixture, Measurements &measurements,
                          std::size_t samples);
[[nodiscard]] bool ObserveAndReport(Fixture &fixture,
                                    Measurements &measurements,
                                    std::size_t samples);

} // namespace rund::measure::compute::nested_repeat
