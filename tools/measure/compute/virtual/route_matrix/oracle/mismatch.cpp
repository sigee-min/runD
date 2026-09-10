#include "local.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::measure::compute::route_matrix::oracle {

using oracle_detail::device_vsm_cohort;
using oracle_detail::window_ring_cohort;

const char *mismatch_reason(const Row &row) noexcept {
  if (!row.spec.supported || !row.cpu_ok || !row.output_ok ||
      !row.backing_match) {
    return !row.spec.supported ? "unsupported"
           : !row.cpu_ok       ? "cpu_output"
           : !row.output_ok    ? "output"
                               : "backing";
  }
  if (row.status != "ok") {
    return "status";
  }
  if (row.backend == ComputeBackend::Unavailable) {
    return "backend";
  }
  if (row.stats.backend != row.backend) {
    return "stats_backend";
  }
  if (row.page_count != row.spec.q || row.residency.page_count != row.spec.q) {
    return row.page_count != row.spec.q ? "page_count" : "residency_page_count";
  }
  if (row.residency.frame_capacity !=
      std::min<std::uint64_t>(row.spec.q, FrameCapacity)) {
    return "frame_capacity";
  }
  if ((row.spec.family == "pointwise_callback" ||
       row.spec.family == "pointwise_resident") &&
      row.route != row.expected_route) {
    return "route";
  }
  if (!row.one_final) {
    return "cohort";
  }
  if (!row.allocation_free_60) {
    return "allocation_free_60";
  }
  if (!row.timing_valid) {
    return "timing";
  }
  if (row.spec.family == "pointwise_resident" && row.route != "device_vsm") {
    return "route";
  }
  if (row.spec.family == "spatial_window") {
    if (row.route != "device_vsm") {
      return "route";
    }
    if (!window_ring_cohort(row)) {
      return "receipt";
    }
    return nullptr;
  }
  if (row.spec.family == "pointwise_callback" && row.route == "ordinary") {
    return "route";
  }
  if (row.spec.family == "pointwise_resident" && !device_vsm_cohort(row)) {
    return "cohort";
  }
  if (row.route == "device_vsm") {
    if (!row.owner_stable) {
      return "owner_stable";
    }
    if (!row.route_evidence.proof_valid) {
      return "proof_valid";
    }
    if (!device_vsm_cohort(row)) {
      return "cohort";
    }
  }
  return nullptr;
}

bool exact_route(const Row &row) noexcept {
  return mismatch_reason(row) == nullptr;
}

} // namespace rund::measure::compute::route_matrix::oracle
