#include "local.hpp"

#include "src/accel/range_aggregate/execution/projection.hpp"

#include <iostream>

namespace node_accel_contract::stencil::backend {

bool StencilMatch(const bool ok, const char *const name) {
  if (ok) {
    return true;
  }
  std::cerr << "stencil backend match failed: " << name << '\n';
  return false;
}

[[nodiscard]] std::optional<rund::node::accel::detail::RangeCandidate>
RangeExecCandidate(const rund::node::accel::detail::RangePlan &plan) noexcept {
  using namespace rund::node::accel::detail;
  const std::optional<RangeExec> execution = RangeExec::from(plan);
  return execution.has_value()
             ? std::optional<RangeCandidate>{execution->plan().candidate()}
             : std::nullopt;
}

bool RuntimeSharedProbeMatchesContract(const RuntimeSharedProbe probe,
                                       const char *const backend) {
  switch (probe.status) {
  case RuntimeSharedProbeStatus::SharedVerified:
    if (probe.candidate.has_value() && probe.candidate->uses_shared_halo() &&
        probe.candidate->radius_capacity() != 0u) {
      return true;
    }
    break;
  case RuntimeSharedProbeStatus::NoSharedCapabilityVerified:
    if (!probe.candidate.has_value()) {
      return true;
    }
    break;
  case RuntimeSharedProbeStatus::Failed:
    break;
  }
  std::cerr << backend << " runtime maximum shared contract failed: status="
            << static_cast<unsigned>(probe.status) << " width="
            << (probe.candidate.has_value() ? probe.candidate->width() : 0u)
            << " cap="
            << (probe.candidate.has_value() ? probe.candidate->radius_capacity()
                                            : 0u)
            << '\n';
  return false;
}

} // namespace node_accel_contract::stencil::backend
