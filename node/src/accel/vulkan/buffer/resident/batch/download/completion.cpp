#include "../../../../../backend/result.hpp"

#include "internal.hpp"

#include <algorithm>

namespace rund::node::accel::detail::batch_download_internal {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

void MarkDownloadChunkFailure(BackendDownload &result,
                              const std::span<const DownloadRoute> requests,
                              const std::span<const std::uint64_t> confirmed,
                              const std::span<const DownloadPlan> chunk_plans,
                              const char *const reason,
                              const bool submitted) noexcept {
  if (submitted) {
    ::rund::detail::counter::Accumulate(result.command_submits, 1u);
  }
  result.check = {false, reason};
  result.payload_hash_valid = false;
  const std::size_t failed = chunk_plans[0u].request;
  MarkDownloadFailure(requests[failed].outcome,
                      confirmed[failed] != 0u
                          ? DownloadRangeState::FailedMayWrite
                          : DownloadRangeState::FailedNoWrite,
                      confirmed[failed]);
  result.first_failed = failed;
  result.first_failed_valid = true;
  result.confirmed_bytes = 0u;
  result.ordered_prefix = 0u;
  for (std::size_t index = 0u; index < requests.size(); ++index) {
    ::rund::detail::counter::Accumulate(result.confirmed_bytes,
                                        confirmed[index]);
    if (index < failed && requests[index].outcome != nullptr &&
        requests[index].outcome->state == DownloadRangeState::Complete) {
      ++result.ordered_prefix;
    }
  }
}

void MarkDownloadStagingFailure(BackendDownload &result,
                                const std::span<const DownloadRoute> requests,
                                const std::span<const std::uint64_t> confirmed,
                                const std::size_t failed,
                                const char *const reason) noexcept {
  result.check = {false, reason};
  result.payload_hash_valid = false;
  MarkDownloadFailure(requests[failed].outcome,
                      DownloadRangeState::FailedNoWrite, confirmed[failed]);
  result.first_failed = failed;
  result.first_failed_valid = true;
}

void MarkDownloadMappedFailure(
    BackendDownload &result, const std::span<const DownloadRoute> requests,
    const std::span<const std::uint64_t> confirmed,
    const std::span<const DownloadPlan> chunk_plans) noexcept {
  result.check = {false, "accel_vulkan_transfer_invalid"};
  result.payload_hash_valid = false;
  const std::size_t failed = chunk_plans[0u].request;
  MarkDownloadFailure(requests[failed].outcome,
                      DownloadRangeState::FailedMayWrite, confirmed[failed]);
  result.first_failed = failed;
  result.first_failed_valid = true;
}

void MarkDownloadWaitFailure(BackendDownload &result,
                             const std::span<const DownloadRoute> requests,
                             const std::size_t failed,
                             const std::uint64_t failed_confirmed,
                             const bool use_outcome_confirmed) noexcept {
  if (failed >= requests.size()) {
    return;
  }
  MarkDownloadFailure(requests[failed].outcome,
                      DownloadRangeState::FailedMayWrite, failed_confirmed);
  for (std::size_t index = failed + 1u; index < requests.size(); ++index) {
    ResetDownloadOutcome(requests[index].outcome);
  }
  result.first_failed = failed;
  result.first_failed_valid = true;
  result.ordered_prefix = use_outcome_confirmed ? 0u : failed;
  result.confirmed_bytes = 0u;
  for (std::size_t index = 0u; index < failed; ++index) {
    if (requests[index].outcome != nullptr &&
        requests[index].outcome->state == DownloadRangeState::Complete) {
      if (use_outcome_confirmed) {
        ++result.ordered_prefix;
        ::rund::detail::counter::Accumulate(
            result.confirmed_bytes, requests[index].outcome->confirmed_bytes);
      } else {
        ::rund::detail::counter::Accumulate(result.confirmed_bytes,
                                            requests[index].bytes);
      }
    }
  }
}

void FinishDownloadSuccess(
    BackendDownload &result,
    const std::span<const DownloadRoute> requests) noexcept {
  result.ordered_prefix = requests.size();
  for (const DownloadRoute &request : requests) {
    ::rund::detail::counter::Accumulate(result.confirmed_bytes, request.bytes);
  }
  result.staging_reused = result.staging_bytes != 0u &&
                          result.staging_reused_bytes == result.staging_bytes;
}

#endif

} // namespace rund::node::accel::detail::batch_download_internal
