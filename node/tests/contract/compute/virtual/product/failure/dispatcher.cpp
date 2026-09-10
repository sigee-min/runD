#include "local.hpp"

namespace rund_node_test_virtual::product {

int CheckProductBackingFailure(const rund::compute::Backend backend) {
  failure::FailureFixture fixture{};
  if (const int initialization =
          failure::InitializeFailureFixture(fixture, backend);
      initialization != 0) {
    return initialization;
  }
  if (const int result = failure::CheckGenericBackingFailure(fixture);
      result != 0) {
    return result;
  }
  if (backend == rund::compute::Backend::Cpu) {
    return 0;
  }
  if (const int result = failure::CheckDeviceVsmFailure(fixture); result != 0) {
    return result;
  }
  if (const int result = failure::CheckNativeTransferFailure(fixture);
      result != 0) {
    return result;
  }
  return failure::CheckMultiLaneFailure(fixture);
}

} // namespace rund_node_test_virtual::product
