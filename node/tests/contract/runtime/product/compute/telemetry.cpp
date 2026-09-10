#include "telemetry/local.hpp"

namespace rund::node::test_contract {

int CheckTelemetry() {
  if (const int result = telemetry_contract::CheckSurfaceAndFindings();
      result != 0) {
    return result;
  }
  if (const int result = telemetry_contract::CheckBasicAndDetail();
      result != 0) {
    return result;
  }
  return telemetry_contract::CheckTraceAndTiming();
}

} // namespace rund::node::test_contract

int RunTelemetryDetailContract() {
  return rund::node::test_contract::CheckTelemetry();
}
