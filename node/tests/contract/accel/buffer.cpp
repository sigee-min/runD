#include <accel/device.hpp>

#include "buffer/local.hpp"
#include "buffer/reject/run.hpp"

namespace node_accel_contract {

bool PublicBufferApiContract(const rund::AccelDevice &pick) {
  return PublicBufferApiRejectsInvalidDescriptors() &&
         PublicBufferApiRejectsUnavailableBackends() &&
         PublicBufferApiExposesMetalResidencyWhenAvailable(pick) &&
         PublicBufferApiRoundTripsAndReportsStatsWhenAvailable(pick) &&
         PublicBufferApiRejectsRangeAndOwnerFailures(pick);
}

} // namespace node_accel_contract
