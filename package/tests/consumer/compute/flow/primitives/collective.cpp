#include "model.hpp"

namespace package_compute {

int Collective() {
  auto reduced =
      rund::compute::on(rund::compute::Target::cpu(), values)
          .reduce(rund::compute::Reduce::Sum)
          .map("scalar-adjust", [](auto value) { return value + 1u; })
          .collect();
  if (!reduced) {
    return reduced.exit_code();
  }
  if (*reduced != std::vector<std::uint32_t>{11u}) {
    return FlowMismatch(__LINE__);
  }

  auto sorted =
      rund::compute::on(rund::compute::Target::cpu(), values).sort().collect();
  if (!sorted) {
    return sorted.exit_code();
  }
  if (*sorted != std::vector<std::uint32_t>{1u, 2u, 3u, 4u}) {
    return FlowMismatch(__LINE__);
  }

  auto gathered = rund::compute::on(rund::compute::Target::cpu(), values)
                      .gather(indices)
                      .collect();
  if (!gathered) {
    return gathered.exit_code();
  }
  if (*gathered != std::vector<std::uint32_t>{1u, 2u, 3u, 4u}) {
    return FlowMismatch(__LINE__);
  }

  auto order = rund::compute::on(rund::compute::Target::cpu(), values)
                   .argsort()
                   .collect();
  if (!order) {
    return order.exit_code();
  }
  if (*order != std::vector<std::uint32_t>{1u, 3u, 0u, 2u}) {
    return FlowMismatch(__LINE__);
  }

  auto scattered = rund::compute::on(rund::compute::Target::cpu(), values)
                       .scatter(indices)
                       .collect();
  if (!scattered) {
    return scattered.exit_code();
  }
  if (*scattered != std::vector<std::uint32_t>{4u, 3u, 2u, 1u}) {
    return FlowMismatch(__LINE__);
  }

  auto scanned = rund::compute::on(rund::compute::Target::cpu(), values)
                     .segmented_scan(heads, rund::compute::Scan::InclusiveSum)
                     .collect();
  if (!scanned) {
    return scanned.exit_code();
  }
  if (*scanned != std::vector<std::uint32_t>{3u, 4u, 4u, 6u}) {
    return FlowMismatch(__LINE__);
  }

  auto segmented = rund::compute::on(rund::compute::Target::cpu(), values)
                       .segmented_reduce(heads)
                       .collect();
  if (!segmented) {
    return segmented.exit_code();
  }
  if (segmented->size() != values.size() || (*segmented)[0] != 4u ||
      (*segmented)[1] != 6u) {
    return FlowMismatch(__LINE__);
  }

  auto histogram = rund::compute::on(rund::compute::Target::cpu(), values)
                       .histogram({.bins = 5u})
                       .collect();
  if (!histogram) {
    return histogram.exit_code();
  }
  if (*histogram != std::vector<std::uint32_t>{0u, 1u, 1u, 1u, 1u}) {
    return FlowMismatch(__LINE__);
  }
  return 0;
}

} // namespace package_compute
