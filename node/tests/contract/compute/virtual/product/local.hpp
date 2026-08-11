#pragma once

#include <rund/compute/backend.hpp>

#include <cstdint>

namespace rund_node_test_virtual::product {

enum class ProductCapability : std::uint8_t {
  Executable,
  BackendBlocked,
  ContractFailure,
};

struct ProductCapabilityResult final {
  ProductCapability disposition{ProductCapability::ContractFailure};
  int detail{};
};

[[nodiscard]] int CheckProductSurface();
[[nodiscard]] ProductCapabilityResult
CheckProductCapability(rund::compute::Backend backend);
[[nodiscard]] int CheckProductPrepare(rund::compute::Backend backend);
[[nodiscard]] int CheckProductExecution(rund::compute::Backend backend);
[[nodiscard]] int CheckProductActiveCount(rund::compute::Backend backend);
[[nodiscard]] int CheckProductBackingFailure(rund::compute::Backend backend);
[[nodiscard]] int CheckProductWidthProjection(rund::compute::Backend backend);
[[nodiscard]] int CheckProductPrefetch(rund::compute::Backend backend);
[[nodiscard]] int CheckProductWindow(rund::compute::Backend backend);
[[nodiscard]] int CheckProductReduce(rund::compute::Backend backend);
[[nodiscard]] int CheckProductScan(rund::compute::Backend backend);
[[nodiscard]] int CheckProductDeviceCache(rund::compute::Backend backend);
[[nodiscard]] int
CheckProductPipelineConcurrency(rund::compute::Backend backend);
[[nodiscard]] int
CheckProductBackingConcurrency(rund::compute::Backend backend);

} // namespace rund_node_test_virtual::product
