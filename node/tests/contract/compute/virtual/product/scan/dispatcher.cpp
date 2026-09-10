#include "support.hpp"

namespace rund_node_test_virtual::product {

int CheckProductScan(const rund::compute::Backend backend) {
  scan::ScanFixture fixture{};
  if (const int result = scan::InitializeScanFixture(fixture, backend);
      result != 0) {
    return result;
  }
  if (const int result = scan::RunBasic(fixture); result != 0) {
    return result;
  }
  if (const int result = scan::BeginTieredRoute(fixture); result != 0) {
    return result;
  }
  if (const int result = scan::RunTieredCold(fixture); result != 0) {
    return result;
  }
  if (const int result = scan::RunOverflow(fixture); result != 0) {
    return result;
  }
  if (const int result = scan::RunTieredWide(fixture); result != 0) {
    return result;
  }
  if (const int result = scan::RunUnknown(fixture); result != 0) {
    return result;
  }
  scan::ResetTieredRoute(fixture);
  return scan::RunPoisonRecovery(fixture);
}

} // namespace rund_node_test_virtual::product
