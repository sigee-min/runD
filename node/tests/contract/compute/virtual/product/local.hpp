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
[[nodiscard]] int CheckProductDeviceVsmUnknown(rund::compute::Backend backend);
[[nodiscard]] int CheckProductDeviceVsmCompletionLifetime();
[[nodiscard]] int CheckProductActiveCount(rund::compute::Backend backend);
[[nodiscard]] int CheckProductBackingFailure(rund::compute::Backend backend);
[[nodiscard]] int CheckProductReturnedFailureBoundary();
[[nodiscard]] int CheckProductWidthProjection(rund::compute::Backend backend);
[[nodiscard]] int CheckProductPrefetch(rund::compute::Backend backend);
[[nodiscard]] int CheckProductWindow(rund::compute::Backend backend);
[[nodiscard]] int CheckProductReduce(rund::compute::Backend backend);
[[nodiscard]] int CheckProductGraphWavefront(rund::compute::Backend backend);
[[nodiscard]] int CheckProductGraphForecastWindow(rund::compute::Backend);
[[nodiscard]] int
CheckProductGraphHostWavefront(rund::compute::Backend backend);
[[nodiscard]] int
CheckProductGraphMultiHostWavefront(rund::compute::Backend backend);
[[nodiscard]] int
CheckProductGraphLaterMultiHostWavefront(rund::compute::Backend backend);
[[nodiscard]] int CheckProductGraphPointwise(rund::compute::Backend backend);
[[nodiscard]] int
CheckProductGraphPointwiseWideHost(rund::compute::Backend backend);
[[nodiscard]] int
CheckProductGraphPointwiseWideDevice(rund::compute::Backend backend);
[[nodiscard]] int
CheckProductGraphPointwiseDeepDevice(rund::compute::Backend backend);
[[nodiscard]] int
CheckProductGraphPointwiseDeeperDevice(rund::compute::Backend backend);
[[nodiscard]] int
CheckProductGraphPointwiseFrontierDevice(rund::compute::Backend backend);
[[nodiscard]] int
CheckProductGraphPointwiseDepthSixDevice(rund::compute::Backend backend);
[[nodiscard]] int
CheckProductGraphPointwiseDepthSevenDevice(rund::compute::Backend backend);
[[nodiscard]] int
CheckProductGraphPointwiseMulti(rund::compute::Backend backend,
                                bool require_adapter = false);
[[nodiscard]] int CheckProductGraphPersistRing(rund::compute::Backend backend);
[[nodiscard]] int CheckProductScan(rund::compute::Backend backend);
[[nodiscard]] int CheckProductDeviceCache(rund::compute::Backend backend);
[[nodiscard]] int
CheckProductPipelineConcurrency(rund::compute::Backend backend);
[[nodiscard]] int
CheckProductBackingConcurrency(rund::compute::Backend backend);

} // namespace rund_node_test_virtual::product
