#include "parity/local.hpp"

int RunRuntimeTaskReplayTelemetryContract() {
  if (const int result = replay_telemetry_contract::CheckParity();
      result != 0) {
    return result;
  }
  return replay_telemetry_contract::CheckFailures();
}
