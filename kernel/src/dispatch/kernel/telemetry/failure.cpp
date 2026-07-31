#include "../local.hpp"

namespace rund::kernel::dispatch::detail {

Result FailResult(const Plan &plan, const char *reason) {
  return Result{
      .ok = false,
      .failure_reason = reason,
      .telemetry = BuildBaseTelemetry(plan),
  };
}

} // namespace rund::kernel::dispatch::detail
