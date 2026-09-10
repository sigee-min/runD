#include "local.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::measure::compute::route_matrix::oracle {

const char *route_name(const ::rund::compute::Stats &stats,
                       const RouteEvidence &route,
                       const std::uint64_t owner_id) noexcept {
  if (owner_id != 0u && route.owner_events != 0u && route.owner_stable &&
      route.proof_seen && route.proof_valid && route.proof_identity_stable &&
      route.proof_flags_stable &&
      (route.proof_hi != 0u || route.proof_lo != 0u)) {
    return "device_vsm";
  }
  if (route.proof_seen || route.owner_events != 0u) {
    return "unknown";
  }
  if (stats.pipeline.residency.window_handoff_count != 0u ||
      stats.pipeline.residency.window_batch_count != 0u) {
    return "persistent";
  }
  if (stats.pipeline.residency.page_count != 0u) {
    return "paged";
  }
  return "ordinary";
}

const char *label(const WarmSummary &cpu, const WarmSummary &backend) noexcept {
  if (!cpu.complete || !backend.complete || cpu.p25_us <= 0.0 ||
      backend.p25_us <= 0.0) {
    return "indeterminate";
  }
  if (backend.p75_us < cpu.p25_us) {
    return "backend_observed";
  }
  if (cpu.p75_us < backend.p25_us) {
    return "cpu_observed";
  }
  return "indeterminate";
}

} // namespace rund::measure::compute::route_matrix::oracle
