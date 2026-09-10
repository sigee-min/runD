#include "model.hpp"

namespace rund::measure::compute::nested_repeat {

bool Sample(Fixture &fixture, Measurements &measurements,
            const std::size_t samples) {
  if (!RunSerial(fixture, measurements, false) ||
      !RunNested(fixture, measurements, nullptr) ||
      !RunRepeated(fixture, measurements, nullptr) ||
      !RunSealed(fixture, measurements, nullptr)) {
    return false;
  }
  if (!RunSerial(fixture, measurements, false) ||
      !RunNested(fixture, measurements, nullptr) ||
      !RunNested(fixture, measurements, nullptr) ||
      !RunSerial(fixture, measurements, false)) {
    return false;
  }
  for (std::size_t sample = 0u; sample < samples; ++sample) {
    const bool serial_first = sample % 2u == 0u;
    measurements.serial_pair_serial_first += serial_first ? 1u : 0u;
    measurements.serial_pair_nested_first += serial_first ? 0u : 1u;
    const bool ok = serial_first
                        ? RunSerial(fixture, measurements, true) &&
                              RunNested(fixture, measurements,
                                        &measurements.serial_pair_nested_wall)
                        : RunNested(fixture, measurements,
                                    &measurements.serial_pair_nested_wall) &&
                              RunSerial(fixture, measurements, true);
    if (!ok) {
      return false;
    }
  }

  if (!RunNested(fixture, measurements, nullptr) ||
      !RunSealed(fixture, measurements, nullptr) ||
      !RunSealed(fixture, measurements, nullptr) ||
      !RunNested(fixture, measurements, nullptr)) {
    return false;
  }
  for (std::size_t sample = 0u; sample < samples; ++sample) {
    const bool nested_first = sample % 2u == 0u;
    measurements.latency_pair_nested_first += nested_first ? 1u : 0u;
    measurements.latency_pair_sealed_first += nested_first ? 0u : 1u;
    const bool ok = nested_first
                        ? RunNested(fixture, measurements,
                                    &measurements.latency_pair_nested_wall) &&
                              RunSealed(fixture, measurements,
                                        &measurements.latency_pair_sealed_wall)
                        : RunSealed(fixture, measurements,
                                    &measurements.latency_pair_sealed_wall) &&
                              RunNested(fixture, measurements,
                                        &measurements.latency_pair_nested_wall);
    if (!ok) {
      return false;
    }
  }

  if (!RunRepeated(fixture, measurements, nullptr) ||
      !RunSealed(fixture, measurements, nullptr) ||
      !RunSealed(fixture, measurements, nullptr) ||
      !RunRepeated(fixture, measurements, nullptr)) {
    return false;
  }
  for (std::size_t sample = 0u; sample < samples; ++sample) {
    const bool repeated_first = sample % 2u == 0u;
    measurements.throughput_pair_repeated_first += repeated_first ? 1u : 0u;
    measurements.throughput_pair_sealed_first += repeated_first ? 0u : 1u;
    const bool ok =
        repeated_first
            ? RunRepeated(fixture, measurements, &measurements.repeated_wall) &&
                  RunSealed(fixture, measurements,
                            &measurements.throughput_pair_sealed_wall)
            : RunSealed(fixture, measurements,
                        &measurements.throughput_pair_sealed_wall) &&
                  RunRepeated(fixture, measurements,
                              &measurements.repeated_wall);
    if (!ok) {
      return false;
    }
  }

  return RunSerial(fixture, measurements, false) &&
         RunNested(fixture, measurements, nullptr) &&
         RunRepeated(fixture, measurements, nullptr) &&
         RunSealed(fixture, measurements, nullptr);
}

} // namespace rund::measure::compute::nested_repeat
