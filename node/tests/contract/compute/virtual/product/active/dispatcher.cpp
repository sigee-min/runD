#include "local.hpp"

namespace rund_node_test_virtual::product {

int CheckProductActiveCount(const rund::compute::Backend backend) {
  active::ActiveFixture fixture{};
  if (const int initialization =
          active::InitializeActiveFixture(fixture, backend);
      initialization != 0) {
    return initialization;
  }
  if (const int base = active::CheckBase(fixture); base != 0) {
    return base;
  }
  if (const int growth = active::CheckGrowth(fixture); growth != 0) {
    return growth;
  }
  return active::CheckCache(fixture);
}

}  // namespace rund_node_test_virtual::product
