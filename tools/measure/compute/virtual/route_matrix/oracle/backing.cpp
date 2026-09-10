#include "local.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::measure::compute::route_matrix::oracle {

namespace oracle_detail {

[[nodiscard]] bool memory_receipt_ok(const Row &row) noexcept {
  if (row.route != "persistent" || row.spec.backing != "memory" ||
      row.spec.q == 0u ||
      row.spec.q > std::numeric_limits<std::uint64_t>::max() / PageElements) {
    return false;
  }
  std::uint64_t logical_elements = row.spec.q * PageElements;
  if (logical_elements == 0u) {
    return false;
  }
  --logical_elements;
  std::uint64_t logical_bytes = 0u;
  if (!mul(logical_elements, sizeof(std::int32_t), logical_bytes)) {
    return false;
  }
  if (row.spec.window >
      (std::numeric_limits<std::uint64_t>::max() - PageElements) / 2u) {
    return false;
  }
  const std::uint64_t frame = PageElements + row.spec.window * 2u;
  std::uint64_t frame_bytes = 0u;
  std::uint64_t page_bytes = 0u;
  if (!mul(frame, sizeof(std::int32_t), frame_bytes) ||
      !mul(row.spec.q, frame_bytes, page_bytes)) {
    return false;
  }
  const auto &stats = row.stats;
  const auto &residency = row.residency;
  const auto &transfers = stats.transfer_submissions;
  const bool zero_transfers =
      transfers.host_to_device == 0u && transfers.device_to_host == 0u &&
      transfers.device_to_device == 0u && stats.uploaded_bytes == 0u &&
      stats.downloaded_bytes == 0u;
  if (residency.page_in_count >
      std::numeric_limits<std::uint64_t>::max() - residency.cache_hit_count) {
    return false;
  }
  const bool supplied_ok =
      residency.page_in_count + residency.cache_hit_count == row.spec.q;
  const bool page_in_ok = residency.page_in_count == 0u
                              ? residency.page_in_bytes == 0u
                              : residency.page_in_bytes == page_bytes;
  const bool read_ok = residency.backing_read_bytes == 0u ||
                       residency.backing_read_bytes == logical_bytes;
  return zero_transfers && supplied_ok && page_in_ok &&
         residency.page_out_count == row.spec.q &&
         residency.page_out_bytes == logical_bytes &&
         residency.backing_write_bytes == logical_bytes && read_ok;
}

[[nodiscard]] bool device_vsm_proof_ok(const Row &row) noexcept {
  const auto &route = row.route_evidence;
  return row.route == "device_vsm" && row.owner_id != 0u &&
         route.owner_events != 0u && route.owner_stable && route.proof_seen &&
         route.proof_valid && route.proof_identity_stable &&
         route.proof_flags_stable &&
         (route.proof_hi != 0u || route.proof_lo != 0u);
}

} // namespace oracle_detail

using oracle_detail::device_vsm_proof_ok;
using oracle_detail::memory_receipt_ok;

bool backing_ok(const Row &row) noexcept {
  const auto &route = row.route_evidence;
  const auto &stats = row.stats;
  if (row.spec.resident) {
    return device_vsm_proof_ok(row) &&
           route.public_gpu_addressable_backing_runs == CohortRuns &&
           route.whole_run_staging_runs == 0u && stats.uploaded_bytes == 0u &&
           stats.downloaded_bytes == 0u;
  }
  if (row.route == "device_vsm") {
    return device_vsm_proof_ok(row) &&
           route.whole_run_staging_runs == CohortRuns &&
           route.public_gpu_addressable_backing_runs == 0u &&
           stats.uploaded_bytes == 0u && stats.downloaded_bytes == 0u;
  }
  return memory_receipt_ok(row);
}

} // namespace rund::measure::compute::route_matrix::oracle
