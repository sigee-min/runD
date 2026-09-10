#include "local.hpp"

namespace rund_node_test_pipeline_residency {

int CheckSlidingAuthority() {
  int result = sliding_authority::CheckPhysicalForecast();
  if (result != 0) {
    return result;
  }
  result = sliding_authority::CheckInactiveBank();
  if (result != 0) {
    return result;
  }
  result = sliding_authority::CheckHaloReuse();
  if (result != 0) {
    return result;
  }
  result = sliding_authority::CheckOutputReservation();
  if (result != 0) {
    return result;
  }
  result = sliding_authority::CheckMalformedReceipts();
  if (result != 0) {
    return result;
  }
  return sliding_authority::CheckCacheAuthority();
}

} // namespace rund_node_test_pipeline_residency
