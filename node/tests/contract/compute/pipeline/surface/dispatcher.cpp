#include "local.hpp"

#include <array>

namespace rund_node_test_pipeline {

[[nodiscard]] int CheckSurface(rund::compute::Device &device) {
  constexpr std::array<std::int32_t, 4u> input_values{1, 2, 3, 4};
  auto input = Upload(device, input_values);
  if (!input) {
    return 1;
  }
  if (const int record = CheckSurfaceRecord(device, *input); record != 0) {
    return record;
  }
  if (const int bounded = CheckSurfaceBounded(device, *input);
      bounded != 0) {
    return bounded;
  }
  if (const int alias = CheckSurfaceAlias(device, *input); alias != 0) {
    return alias;
  }
  return CheckSurfaceSharedCount(device);
}

} // namespace rund_node_test_pipeline
