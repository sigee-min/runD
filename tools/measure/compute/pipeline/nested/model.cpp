#include "model.hpp"

namespace rund::measure::compute::nested_repeat {

void Measurements::reserve(const std::size_t samples) {
  serial_wall.reserve(samples);
  serial_pair_nested_wall.reserve(samples);
  latency_pair_nested_wall.reserve(samples);
  latency_pair_sealed_wall.reserve(samples);
  repeated_wall.reserve(samples);
  throughput_pair_sealed_wall.reserve(samples);
}

} // namespace rund::measure::compute::nested_repeat
