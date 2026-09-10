#include "local.hpp"

namespace rund_node_test_virtual::product {

int CheckProductReduce(const rund::compute::Backend backend) {
  reduce::ReduceFixture fixture{};
  const int initialized = reduce::InitializeReduceFixture(fixture, backend);
  if (initialized != 0 || fixture.backend_unavailable) {
    return initialized;
  }
  if (const int result = reduce::RunBasicReduce(fixture, backend);
      result != 0) {
    return result;
  }
  if (const int result = reduce::RunSequenceReduce(fixture, backend);
      result != 0) {
    return result;
  }
  if (const int result = reduce::RunLendingReduce(fixture, backend);
      result != 0) {
    return result;
  }
  return reduce::RunDirectReduce(fixture, backend);
}

} // namespace rund_node_test_virtual::product
