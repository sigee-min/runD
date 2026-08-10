#include "model.hpp"

#include <algorithm>

namespace rund::measure::compute::virtual_residency {

void WallSamples::sort() noexcept {
  std::sort(microseconds.begin(), microseconds.end());
}

double WallSamples::p50() const noexcept {
  constexpr std::size_t middle = WarmSamples / 2u;
  return microseconds[middle - 1u] +
         (microseconds[middle] - microseconds[middle - 1u]) / 2.0;
}

double WallSamples::p95() const noexcept {
  return microseconds[nearest_rank_index(WarmSamples, 95u)];
}

} // namespace rund::measure::compute::virtual_residency
