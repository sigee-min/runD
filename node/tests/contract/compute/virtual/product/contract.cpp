#include "local.hpp"

#include "../../../target/selection.hpp"

#include <cstdio>

int RunComputeVirtualResidencyProductContract() {
  using namespace rund_node_test_virtual::product;
  if (CheckProductDeviceVsmCompletionLifetime() != 0) {
    std::fprintf(stderr, "compute virtual product completion lifetime\n");
    return 98;
  }
  if (CheckProductReturnedFailureBoundary() != 0) {
    std::fprintf(stderr, "compute virtual product returned failure boundary\n");
    return 99;
  }
  if (const int result = CheckProductSurface(); result != 0) {
    std::fprintf(stderr, "compute virtual product surface result=%d\n", result);
    return 100 + result;
  }
  for (const rund::compute::Backend backend :
       rund::node::test_contract::selected_compute_backends()) {
    const ProductCapabilityResult capability = CheckProductCapability(backend);
    if (capability.disposition == ProductCapability::ContractFailure) {
      std::fprintf(stderr,
                   "compute virtual product capability backend=%u result=%d\n",
                   static_cast<unsigned>(backend), capability.detail);
      return static_cast<int>(backend) * 1000 + 50 + capability.detail;
    }
    if (capability.disposition == ProductCapability::BackendBlocked) {
      std::fprintf(stderr,
                   "compute virtual product status=blocked backend=%u "
                   "reason=compute_backend_unsupported\n",
                   static_cast<unsigned>(backend));
      continue;
    }
    if (const int result = CheckProductPrepare(backend); result != 0) {
      std::fprintf(stderr,
                   "compute virtual product prepare backend=%u result=%d\n",
                   static_cast<unsigned>(backend), result);
      return static_cast<int>(backend) * 1000 + result;
    }
    if (const int result = CheckProductBackingFailure(backend); result != 0) {
      std::fprintf(stderr,
                   "compute virtual product retry backend=%u result=%d\n",
                   static_cast<unsigned>(backend), result);
      return static_cast<int>(backend) * 1000 + 100 + result;
    }
    if (const int result = CheckProductActiveCount(backend); result != 0) {
      std::fprintf(stderr,
                   "compute virtual product active backend=%u result=%d\n",
                   static_cast<unsigned>(backend), result);
      return static_cast<int>(backend) * 1000 + 125 + result;
    }
    if (const int result = CheckProductWidthProjection(backend); result != 0) {
      std::fprintf(stderr,
                   "compute virtual product width backend=%u result=%d\n",
                   static_cast<unsigned>(backend), result);
      return static_cast<int>(backend) * 1000 + 150 + result;
    }
    if (const int result = CheckProductPrefetch(backend); result != 0) {
      std::fprintf(stderr,
                   "compute virtual product prefetch backend=%u result=%d\n",
                   static_cast<unsigned>(backend), result);
      return static_cast<int>(backend) * 1000 + 160 + result;
    }
    if (const int result = CheckProductWindow(backend); result != 0) {
      std::fprintf(stderr,
                   "compute virtual product window backend=%u result=%d\n",
                   static_cast<unsigned>(backend), result);
      return static_cast<int>(backend) * 1000 + 170 + result;
    }
    if (const int result = CheckProductReduce(backend); result != 0) {
      std::fprintf(stderr,
                   "compute virtual product reduce backend=%u result=%d\n",
                   static_cast<unsigned>(backend), result);
      return static_cast<int>(backend) * 1000 + 172 + result;
    }
    if (const int result = CheckProductGraphWavefront(backend); result != 0) {
      std::fprintf(stderr,
                   "compute virtual product Graph wavefront backend=%u "
                   "result=%d\n",
                   static_cast<unsigned>(backend), result);
      return static_cast<int>(backend) * 1000 + 172 + result;
    }
    if (const int result = CheckProductGraphHostWavefront(backend);
        result != 0) {
      std::fprintf(stderr,
                   "compute virtual product Host Graph wavefront backend=%u "
                   "result=%d\n",
                   static_cast<unsigned>(backend), result);
      return static_cast<int>(backend) * 1000 + 172 + result;
    }
    if (const int result = CheckProductGraphMultiHostWavefront(backend);
        result != 0) {
      std::fprintf(stderr,
                   "compute virtual product multi Host Graph wavefront "
                   "backend=%u result=%d\n",
                   static_cast<unsigned>(backend), result);
      return static_cast<int>(backend) * 1000 + 172 + result;
    }
    if (const int result = CheckProductGraphLaterMultiHostWavefront(backend);
        result != 0) {
      std::fprintf(stderr,
                   "compute virtual product later multi Host Graph "
                   "wavefront backend=%u result=%d\n",
                   static_cast<unsigned>(backend), result);
      return static_cast<int>(backend) * 1000 + 172 + result;
    }
    if (const int result = CheckProductGraphPointwise(backend); result != 0) {
      std::fprintf(stderr,
                   "compute virtual product Graph pointwise backend=%u "
                   "result=%d\n",
                   static_cast<unsigned>(backend), result);
      return static_cast<int>(backend) * 1000 + 172 + result;
    }
    if (const int result = CheckProductGraphPointwiseWideHost(backend);
        result != 0) {
      std::fprintf(stderr,
                   "compute virtual product wide-input Graph pointwise "
                   "backend=%u result=%d\n",
                   static_cast<unsigned>(backend), result);
      return static_cast<int>(backend) * 1000 + 172 + result;
    }
    if (const int result = CheckProductGraphPointwiseWideDevice(backend);
        result != 0) {
      std::fprintf(stderr,
                   "compute virtual product wide-input GPU Graph pointwise "
                   "backend=%u result=%d\n",
                   static_cast<unsigned>(backend), result);
      return static_cast<int>(backend) * 1000 + 172 + result;
    }
    if (const int result = CheckProductGraphPointwiseDeepDevice(backend);
        result != 0) {
      std::fprintf(stderr,
                   "compute virtual product deep GPU Graph pointwise "
                   "backend=%u result=%d\n",
                   static_cast<unsigned>(backend), result);
      return static_cast<int>(backend) * 1000 + 172 + result;
    }
    if (const int result = CheckProductGraphPointwiseDeeperDevice(backend);
        result != 0) {
      std::fprintf(stderr,
                   "compute virtual product deeper GPU Graph pointwise "
                   "backend=%u result=%d\n",
                   static_cast<unsigned>(backend), result);
      return static_cast<int>(backend) * 1000 + 172 + result;
    }
    if (const int result = CheckProductGraphPointwiseFrontierDevice(backend);
        result != 0) {
      std::fprintf(stderr,
                   "compute virtual product frontier GPU Graph pointwise "
                   "backend=%u result=%d\n",
                   static_cast<unsigned>(backend), result);
      return static_cast<int>(backend) * 1000 + 172 + result;
    }
    if (const int result = CheckProductGraphPointwiseDepthSixDevice(backend);
        result != 0) {
      std::fprintf(stderr,
                   "compute virtual product six-stage GPU Graph pointwise "
                   "backend=%u result=%d\n",
                   static_cast<unsigned>(backend), result);
      return static_cast<int>(backend) * 1000 + 172 + result;
    }
    if (const int result = CheckProductGraphPointwiseDepthSevenDevice(backend);
        result != 0) {
      std::fprintf(stderr,
                   "compute virtual product seven-stage GPU Graph pointwise "
                   "backend=%u result=%d\n",
                   static_cast<unsigned>(backend), result);
      return static_cast<int>(backend) * 1000 + 172 + result;
    }
    if (const int result = CheckProductGraphPointwiseMulti(backend);
        result != 0) {
      std::fprintf(stderr,
                   "compute virtual product multi-input Graph pointwise "
                   "backend=%u result=%d\n",
                   static_cast<unsigned>(backend), result);
      return static_cast<int>(backend) * 1000 + 172 + result;
    }
    if (const int result = CheckProductScan(backend); result != 0) {
      std::fprintf(stderr,
                   "compute virtual product scan backend=%u result=%d\n",
                   static_cast<unsigned>(backend), result);
      return static_cast<int>(backend) * 1000 + 173 + result;
    }
    if (const int result = CheckProductDeviceCache(backend); result != 0) {
      std::fprintf(stderr,
                   "compute virtual product cache backend=%u result=%d\n",
                   static_cast<unsigned>(backend), result);
      return static_cast<int>(backend) * 1000 + 174 + result;
    }
    if (const int result = CheckProductPipelineConcurrency(backend);
        result != 0) {
      std::fprintf(stderr,
                   "compute virtual product pipeline concurrency backend=%u "
                   "result=%d\n",
                   static_cast<unsigned>(backend), result);
      return static_cast<int>(backend) * 1000 + 175 + result;
    }
    if (const int result = CheckProductBackingConcurrency(backend);
        result != 0) {
      std::fprintf(stderr,
                   "compute virtual product backing concurrency backend=%u "
                   "result=%d\n",
                   static_cast<unsigned>(backend), result);
      return static_cast<int>(backend) * 1000 + 185 + result;
    }
    if (const int result = CheckProductExecution(backend); result != 0) {
      std::fprintf(stderr,
                   "compute virtual product execution backend=%u result=%d\n",
                   static_cast<unsigned>(backend), result);
      return static_cast<int>(backend) * 1000 + 200 + result;
    }
    // The explicit DeviceVsm unknown-device contract is intentionally last:
    // it closes the exact Device for every following VSM owner without
    // changing the ordinary callback-backed default route.
    if (const int result = CheckProductDeviceVsmUnknown(backend); result != 0) {
      std::fprintf(stderr,
                   "compute virtual product DeviceVsm Unknown backend=%u "
                   "result=%d\n",
                   static_cast<unsigned>(backend), result);
      return static_cast<int>(backend) * 1000 + 220 + result;
    }
  }
  return 0;
}
