#include "authority/local.hpp"

namespace rund_node_test_pipeline_residency::service_free_direct_test {

int CheckAuthority() {
  using namespace authority_detail;
  for (const std::uint64_t iterations : {5u, 9u, 257u}) {
    if (!SuccessCase(iterations)) {
      return 1;
    }
  }
  if (!KnownRetryCase()) {
    return 2;
  }
  if (!KnownPartialRetryCase()) {
    return 3;
  }
  if (!KnownNoWritePartialRetryCase()) {
    return 4;
  }
  if (!FrozenBusyCase()) {
    return 5;
  }
  return UnknownAndForgeryCase() ? 0 : 6;
}

} // namespace rund_node_test_pipeline_residency::service_free_direct_test
