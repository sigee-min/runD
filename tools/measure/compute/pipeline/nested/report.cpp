#include "model.hpp"

#include <array>
#include <limits>
#include <numeric>

namespace rund::measure::compute::nested_repeat {

bool ObserveAndReport(Fixture &fixture, Measurements &measurements,
                      const std::size_t samples) {
  const auto observed = fixture.observe_program.run(fixture.serial_first,
                                                    fixture.serial_observed);
  std::vector<std::uint32_t> serial_values(Outer);
  if (!observed || !observed->read(fixture.serial_observed,
                                   std::span<std::uint32_t>{serial_values})) {
    std::fprintf(stderr, "window repeat %s serial observation failed\n",
                 Name(fixture.backend));
    return false;
  }
  std::array<std::uint32_t, 1u> nested_value{};
  std::array<std::uint32_t, 1u> repeated_value{};
  std::array<std::uint32_t, 1u> sealed_value{};
  if (!fixture.nested.read(fixture.nested_output,
                           std::span<std::uint32_t>{nested_value})) {
    std::fprintf(stderr, "window repeat %s nested observation failed\n",
                 Name(fixture.backend));
    return false;
  }
  if (!fixture.repeated.read(fixture.repeated_output,
                             std::span<std::uint32_t>{repeated_value})) {
    std::fprintf(stderr, "window repeat %s repeated observation failed\n",
                 Name(fixture.backend));
    return false;
  }
  if (!fixture.sealed.read(fixture.sealed_output,
                           std::span<std::uint32_t>{sealed_value})) {
    std::fprintf(stderr, "window repeat %s sealed observation failed\n",
                 Name(fixture.backend));
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
      measurements.serial_pair_serial_first == samples / 2u &&
      measurements.serial_pair_nested_first == samples / 2u &&
      measurements.latency_pair_nested_first == samples / 2u &&
      measurements.latency_pair_sealed_first == samples / 2u &&
      measurements.throughput_pair_repeated_first == samples / 2u &&
      measurements.throughput_pair_sealed_first == samples / 2u;
  const bool parity =
      serial_result == fixture.expected &&
      nested_value[0] == fixture.expected &&
      repeated_value[0] == fixture.expected &&
      sealed_value[0] == fixture.expected && serial_result == nested_value[0] &&
      serial_result == repeated_value[0] && serial_result == sealed_value[0];
  constexpr bool serial_fallback = false;
  constexpr bool nested_fallback = false;
  constexpr bool repeated_fallback = false;
  constexpr bool sealed_fallback = false;
  const bool warm_zero =
      measurements.serial_warm.zero() && measurements.nested_warm.zero() &&
      measurements.repeated_warm.zero() && measurements.sealed_warm.zero();
  const bool contract =
      balanced && parity && warm_zero && !serial_fallback && !nested_fallback &&
      !repeated_fallback && !sealed_fallback &&
      measurements.repeated_counters.command_submits ==
          Repetitions * measurements.nested_counters.command_submits &&
      measurements.repeated_counters.dispatches ==
          Repetitions * measurements.nested_counters.dispatches &&
      measurements.sealed_counters.command_submits ==
          measurements.nested_counters.command_submits &&
      measurements.sealed_counters.dispatches ==
          measurements.nested_counters.dispatches;

  const double serial_us = Median(measurements.serial_wall);
  const double serial_pair_nested_us =
      Median(measurements.serial_pair_nested_wall);
  const double latency_pair_nested_us =
      Median(measurements.latency_pair_nested_wall);
  const double latency_pair_sealed_us =
      Median(measurements.latency_pair_sealed_wall);
  const double repeated_us = Median(measurements.repeated_wall);
  const double throughput_pair_sealed_us =
      Median(measurements.throughput_pair_sealed_wall);
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

  std::printf("window_repeat,%s,%s,%s,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,"
              "%zu,%zu,%zu,%zu,%zu",
              Name(fixture.backend), CommandPath(fixture.backend),
              contract ? "ok" : "contract_failed", Maximum, Outer, Tile, Inner,
              samples, measurements.serial_pair_serial_first,
              measurements.serial_pair_nested_first,
              measurements.latency_pair_nested_first,
              measurements.latency_pair_sealed_first,
              measurements.throughput_pair_repeated_first,
              measurements.throughput_pair_sealed_first, Templates, Commands,
              Repetitions);
  std::printf(",%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.6f,%.6f,%.6f,%.6f", serial_us,
              serial_pair_nested_us, latency_pair_nested_us,
              latency_pair_sealed_us, repeated_us, throughput_pair_sealed_us,
              sealed_equivalent_us, speedup, sealed_throughput_speedup,
              sealed_single_execution_ratio);
  std::printf(
      ",%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu",
      static_cast<unsigned long long>(
          measurements.serial_counters.command_submits),
      static_cast<unsigned long long>(
          measurements.nested_counters.command_submits),
      static_cast<unsigned long long>(
          measurements.repeated_counters.command_submits),
      static_cast<unsigned long long>(
          measurements.sealed_counters.command_submits),
      static_cast<unsigned long long>(measurements.serial_counters.dispatches),
      static_cast<unsigned long long>(measurements.nested_counters.dispatches),
      static_cast<unsigned long long>(
          measurements.repeated_counters.dispatches),
      static_cast<unsigned long long>(measurements.sealed_counters.dispatches),
      static_cast<unsigned long long>(
          measurements.nested_stats.pipeline.control_command_count));
  std::printf(
      ",%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu",
      static_cast<unsigned long long>(
          measurements.serial_warm.buffer_allocations),
      static_cast<unsigned long long>(
          measurements.nested_warm.buffer_allocations),
      static_cast<unsigned long long>(
          measurements.repeated_warm.buffer_allocations),
      static_cast<unsigned long long>(
          measurements.sealed_warm.buffer_allocations),
      static_cast<unsigned long long>(measurements.serial_warm.uploaded_bytes),
      static_cast<unsigned long long>(measurements.nested_warm.uploaded_bytes),
      static_cast<unsigned long long>(
          measurements.repeated_warm.uploaded_bytes),
      static_cast<unsigned long long>(measurements.sealed_warm.uploaded_bytes));
  std::printf(
      ",%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu",
      static_cast<unsigned long long>(measurements.serial_warm.download_events),
      static_cast<unsigned long long>(measurements.nested_warm.download_events),
      static_cast<unsigned long long>(
          measurements.repeated_warm.download_events),
      static_cast<unsigned long long>(measurements.sealed_warm.download_events),
      static_cast<unsigned long long>(
          measurements.serial_warm.downloaded_bytes),
      static_cast<unsigned long long>(
          measurements.nested_warm.downloaded_bytes),
      static_cast<unsigned long long>(
          measurements.repeated_warm.downloaded_bytes),
      static_cast<unsigned long long>(
          measurements.sealed_warm.downloaded_bytes));
  std::printf(",%u,%u,%u,%u,%u,%u,%u,%u,%u,%u\n", serial_fallback ? 1u : 0u,
              nested_fallback ? 1u : 0u, repeated_fallback ? 1u : 0u,
              sealed_fallback ? 1u : 0u, serial_result, nested_value[0],
              repeated_value[0], sealed_value[0], parity ? 1u : 0u,
              warm_zero ? 1u : 0u);
  return contract;
}

} // namespace rund::measure::compute::nested_repeat
