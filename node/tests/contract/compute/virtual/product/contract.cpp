#include "local.hpp"

#include "../../../target/selection.hpp"

#include <cstdio>

int RunComputeVirtualResidencyProductContract() {
  using namespace rund_node_test_virtual::product;
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
  }
  return 0;
}
