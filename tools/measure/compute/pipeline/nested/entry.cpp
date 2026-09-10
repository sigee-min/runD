#include "../../pipeline.hpp"
#include "model.hpp"

namespace rund::measure::compute {

bool MeasureNestedRepeat(const Backend backend, const std::size_t samples) {
  if ((backend != Backend::Metal && backend != Backend::Vulkan) ||
      samples == 0u || samples % 4u != 0u) {
    std::fprintf(stderr, "window repeat measurement configuration invalid\n");
    return false;
  }

  auto fixture = nested_repeat::Prepare(backend);
  if (!fixture) {
    return false;
  }
  nested_repeat::Measurements measurements{};
  measurements.reserve(samples);
  return nested_repeat::Sample(*fixture, measurements, samples) &&
         nested_repeat::ObserveAndReport(*fixture, measurements, samples);
}

} // namespace rund::measure::compute
