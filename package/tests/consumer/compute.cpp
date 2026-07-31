#include "compute/telemetry.hpp"

#include "compute/backend.hpp"
#include "compute/batch.hpp"
#include "compute/contracts.hpp"
#include "compute/fixed.hpp"
#include "compute/flow.hpp"
#include "compute/flow/primitives.hpp"
#include "compute/graph/services.hpp"
#include "compute/map.hpp"
#include "compute/memory.hpp"
#include "compute/node/host.hpp"
#include "compute/reuse.hpp"
#include "compute/write.hpp"

#include <cstdio>

namespace {

[[nodiscard]] int Fail(const char *const contract, const int code) {
  std::fprintf(stderr, "package compute contract failed: %s (%d)\n", contract,
               code);
  return code;
}

} // namespace

int main() {
  if (!ComputeStatusSuccess.error().empty() ||
      ComputeStatusBindingFailure.error() != "compute_shape_mismatch" ||
      ComputeStatusInvalidReason.error() != "compute_reason_invalid") {
    return Fail("status", 2);
  }
  if (const int result = package_compute::BackendSurface(); result != 0) {
    return Fail("backend", result);
  }
  if (const int result = package_compute::Map(); result != 0) {
    return Fail("map", result);
  }
  if (const int result = package_compute::BatchRun(); result != 0) {
    return Fail("batch", result);
  }
  if (const int result = package_compute::Reuse(); result != 0) {
    return Fail("reuse", result);
  }
  if (const int result = package_compute::NodeHost(); result != 0) {
    return Fail("node-host", result);
  }
  if (const int result = package_compute::Flow(); result != 0) {
    return Fail("flow", result);
  }
  if (const int result = package_compute::Fixed(); result != 0) {
    return Fail("fixed", result);
  }
  if (const int result = package_compute::FlowPrimitives(); result != 0) {
    return Fail("flow-primitives", result);
  }
  if (const int result = package_compute::Write(); result != 0) {
    return Fail("write", result);
  }
  if (const int result = package_compute::Memory(); result != 0) {
    return Fail("memory", result);
  }
  if (const int result = package_compute::Services(); result != 0) {
    return Fail("services", result);
  }
  if (const int result = package_compute::Telemetry(); result != 0) {
    return Fail("telemetry", result);
  }
  return 0;
}
