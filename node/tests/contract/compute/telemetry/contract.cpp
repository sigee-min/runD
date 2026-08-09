#include "local.hpp"

int RunComputeTelemetryContract() {
  if (!rund_node_test_telemetry::CheckProjection()) {
    return 1;
  }
  if (!rund_node_test_telemetry::CheckProfiles()) {
    return 2;
  }
  return rund_node_test_telemetry::CheckSubmissionAction() ? 0 : 3;
}
