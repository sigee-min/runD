#include "evidence.hpp"

#include "oracle/internal.hpp"

namespace rund_node_test_virtual::product {

namespace {

[[nodiscard]] bool
route_evidence_matches(const ProductExecutionEvidence &evidence) noexcept {
  const auto &run = evidence.final_run;
  if (run.backend == rund::compute::Backend::Cpu) {
    return evidence.owner_mask == 0u && evidence.accepted_owner_count == 0u &&
           evidence.route_kind == RouteKind::CpuRolling &&
           ClassifyMode(run.backend, run.pipeline.residency,
                        run.pipeline.residency.page_count) ==
               evidence.route_kind;
  }
  std::uint32_t expected = 0u;
  switch (evidence.route_kind) {
  case RouteKind::AccelRolling:
    expected = OwnerAccelRolling;
    break;
  case RouteKind::Persistent:
    expected = OwnerPersistent;
    break;
  case RouteKind::DeviceVsm:
    expected = OwnerDeviceVsm;
    break;
  case RouteKind::CpuRolling:
  case RouteKind::Window:
  case RouteKind::ServiceFreeDirect:
  case RouteKind::Unknown:
  case RouteKind::Rejected:
    return false;
  }
  return evidence.owner_mask == expected &&
         evidence.accepted_owner_count == TotalRuns &&
         ClassifyMode(run.backend, run.pipeline.residency,
                      run.pipeline.residency.page_count) == evidence.route_kind;
}

} // namespace

bool ProductExecutionMatches(
    const ProductExecutionEvidence &evidence) noexcept {
  if (!route_evidence_matches(evidence) ||
      !ProductExecutionCommonMatches(evidence)) {
    return false;
  }
  switch (evidence.route_kind) {
  case RouteKind::CpuRolling:
  case RouteKind::AccelRolling:
    return ProductExecutionRollingMatches(evidence);
  case RouteKind::Persistent:
    return ProductExecutionPersistentMatches(evidence);
  case RouteKind::DeviceVsm:
    return ProductExecutionDeviceVsmMatches(evidence);
  case RouteKind::ServiceFreeDirect:
    return false;
  case RouteKind::Window:
  case RouteKind::Unknown:
  case RouteKind::Rejected:
    return ProductExecutionWindowMatches(evidence);
  }
  return false;
}

} // namespace rund_node_test_virtual::product
