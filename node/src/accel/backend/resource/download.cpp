#include "../resource.hpp"

#include "local.hpp"

#include <rund/counter.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] bool ValidDownloadRoute(const DownloadRoute &route) noexcept {
  return route.failure_reason == nullptr && route.resident.id != 0u &&
         route.resident.bytes != 0u && route.handle != nullptr &&
         (route.bytes == 0u || route.data != nullptr) &&
         route.offset <= route.resident.bytes &&
         route.bytes <= route.resident.bytes - route.offset;
}

[[nodiscard]] const char *
DownloadRouteReason(const DownloadRoute &route) noexcept {
  if (route.failure_reason != nullptr) {
    return route.failure_reason;
  }
  if (route.resident.id == 0u || route.resident.bytes == 0u ||
      route.handle == nullptr || (route.bytes != 0u && route.data == nullptr)) {
    return "accel_buffer_unavailable";
  }
  return "accel_buffer_download_overflow";
}

[[nodiscard]] bool Overlap(const DownloadRoute &left,
                           const DownloadRoute &right) noexcept {
  if (left.bytes == 0u || right.bytes == 0u || left.data == nullptr ||
      right.data == nullptr) {
    return false;
  }
  const std::uintptr_t left_begin = reinterpret_cast<std::uintptr_t>(left.data);
  const std::uintptr_t right_begin =
      reinterpret_cast<std::uintptr_t>(right.data);
  if (left.bytes > std::numeric_limits<std::uintptr_t>::max() - left_begin ||
      right.bytes > std::numeric_limits<std::uintptr_t>::max() - right_begin) {
    return true;
  }
  return left_begin < right_begin + right.bytes &&
         right_begin < left_begin + left.bytes;
}

void AddDownload(BackendDownload &total, const BackendDownload &part) noexcept {
  ::rund::detail::counter::Accumulate(total.staging_bytes, part.staging_bytes);
  total.staging_peak_bytes =
      std::max(total.staging_peak_bytes, part.staging_peak_bytes);
  ::rund::detail::counter::Accumulate(total.staging_reused_bytes,
                                      part.staging_reused_bytes);
  ::rund::detail::counter::Accumulate(total.buffer_allocations,
                                      part.buffer_allocations);
  ::rund::detail::counter::Accumulate(total.buffer_reuses, part.buffer_reuses);
  ::rund::detail::counter::Accumulate(total.command_submits,
                                      part.command_submits);
  ::rund::detail::counter::Accumulate(total.readback_ns, part.readback_ns);
  total.staging_reused = total.staging_bytes != 0u &&
                         total.staging_reused_bytes == total.staging_bytes;
}

[[nodiscard]] bool
ExplicitFailure(const DownloadRangeOutcome *const outcome) noexcept {
  return outcome != nullptr &&
         (outcome->state == DownloadRangeState::FailedNoWrite ||
          outcome->state == DownloadRangeState::FailedMayWrite);
}

void SetDownloadPrefix(BackendDownload &result,
                       const std::span<const DownloadRoute> requests,
                       const std::size_t stop) noexcept {
  result.ordered_prefix = 0u;
  result.confirmed_bytes = 0u;
  for (std::size_t index = 0u; index < stop; ++index) {
    const DownloadRangeOutcome *const outcome = requests[index].outcome;
    if (outcome == nullptr || outcome->state != DownloadRangeState::Complete) {
      break;
    }
    ++result.ordered_prefix;
    ::rund::detail::counter::Accumulate(result.confirmed_bytes,
                                        outcome->confirmed_bytes);
  }
}

BackendDownload DownloadScalar(const std::shared_ptr<PickToken> &token,
                               const std::span<const DownloadRoute> requests) {
  BackendDownload total{
      .check = rund::AccelCheck{true, "ok"},
      .payload_hash_valid = true,
  };
  for (std::size_t index = 0u; index < requests.size(); ++index) {
    const DownloadRoute &request = requests[index];
    if (!ValidDownloadRoute(request) || token->ops->download == nullptr) {
      MarkDownloadFailure(request.outcome, DownloadRangeState::FailedNoWrite);
      total.check = {false, token->ops->download == nullptr
                                ? "accel_buffer_backend_unavailable"
                                : DownloadRouteReason(request)};
      total.payload_hash_valid = false;
      total.first_failed = index;
      total.first_failed_valid = true;
      return total;
    }
    const BackendDownload part =
        token->ops->download(token->raw, request.resident, request.handle,
                             request.data, request.bytes, request.offset, true);
    AddDownload(total, part);
    if (!part.check.ok || !part.payload_hash_valid) {
      MarkDownloadFailure(request.outcome,
                          part.confirmed_bytes != 0u
                              ? DownloadRangeState::FailedMayWrite
                              : DownloadRangeState::FailedNoWrite,
                          part.confirmed_bytes);
      total.check = part.check.ok
                        ? rund::AccelCheck{false, "accel_buffer_unavailable"}
                        : part.check;
      total.payload_hash_valid = false;
      ::rund::detail::counter::Accumulate(total.confirmed_bytes,
                                          part.confirmed_bytes);
      total.first_failed = index;
      total.first_failed_valid = true;
      return total;
    }
    if (request.payload_hash != nullptr) {
      *request.payload_hash = part.payload_hash;
    }
    MarkDownloadComplete(request.outcome, request.bytes, part.payload_hash,
                         part.payload_hash_valid);
    ::rund::detail::counter::Accumulate(total.confirmed_bytes, request.bytes);
    ++total.ordered_prefix;
  }
  return total;
}

} // namespace

BackendDownload DownloadBackendBuffer(const std::shared_ptr<PickToken> &token,
                                      const rund::Buffer &buffer,
                                      void *const data,
                                      const std::uint64_t bytes,
                                      const std::uint64_t offset,
                                      const bool hash_payload) {
  const rund::AccelCheck check = resource_detail::Validate(
      token, buffer, data, bytes, offset, "accel_buffer_download_overflow");
  if (!check.ok || token->ops->download == nullptr) {
    return BackendDownload{
        .check = check.ok ? rund::AccelCheck{false,
                                             "accel_buffer_backend_unavailable"}
                          : check};
  }
  return DownloadBackendBuffer(token, resource_detail::Resident(buffer),
                               buffer.handle, data, bytes, offset,
                               hash_payload);
}

BackendDownload
DownloadBackendBuffer(const std::shared_ptr<PickToken> &token,
                      const rund::kernel::ResidentBufferRef &resident,
                      const std::shared_ptr<void> &handle, void *const data,
                      const std::uint64_t bytes, const std::uint64_t offset,
                      const bool hash_payload) {
  if (!resource_detail::ValidRoute(token) || token->ops->download == nullptr ||
      resident.id == 0u || resident.bytes == 0u || handle == nullptr ||
      (bytes != 0u && data == nullptr) || offset > resident.bytes ||
      bytes > resident.bytes - offset) {
    return {};
  }
  return token->ops->download(token->raw, resident, handle, data, bytes, offset,
                              hash_payload);
}

BackendDownload
DownloadBackendBuffers(const std::shared_ptr<PickToken> &token,
                       const std::span<const DownloadRoute> requests,
                       const TransferAuthority authority) {
  if (!resource_detail::ValidRoute(token) || requests.empty()) {
    return {};
  }
  for (const DownloadRoute &request : requests) {
    ResetDownloadOutcome(request.outcome);
  }
  bool eligible = token->ops->download_batch != nullptr &&
                  requests.size() <= kInlineTransferCapacity;
  if (eligible) {
    for (std::size_t index = 0u; index < requests.size(); ++index) {
      if (!ValidDownloadRoute(requests[index])) {
        eligible = false;
        break;
      }
      for (std::size_t prior = 0u; prior < index; ++prior) {
        if (Overlap(requests[prior], requests[index])) {
          eligible = false;
          break;
        }
      }
      if (!eligible) {
        break;
      }
    }
  }
  if (!eligible) {
    return DownloadScalar(token, requests);
  }
  BackendDownload result =
      token->ops->download_batch(token->raw, requests, authority);
  std::size_t explicit_failed = requests.size();
  for (std::size_t index = 0u; index < requests.size(); ++index) {
    if (ExplicitFailure(requests[index].outcome)) {
      explicit_failed = index;
      break;
    }
  }
  if (explicit_failed != requests.size()) {
    if (result.check.ok) {
      result.check = {false, "accel_buffer_backend_unavailable"};
    }
    result.payload_hash_valid = false;
    result.first_failed = explicit_failed;
    result.first_failed_valid = true;
    SetDownloadPrefix(result, requests, explicit_failed);
    return result;
  }
  if (result.check.ok) {
    for (const DownloadRoute &request : requests) {
      if (request.outcome != nullptr &&
          request.outcome->state == DownloadRangeState::Untouched) {
        const std::uint64_t hash =
            request.payload_hash == nullptr ? 0u : *request.payload_hash;
        MarkDownloadComplete(request.outcome, request.bytes, hash,
                             request.payload_hash != nullptr);
      }
    }
    if (result.ordered_prefix == 0u) {
      result.ordered_prefix = requests.size();
    }
    if (result.confirmed_bytes == 0u) {
      for (const DownloadRoute &request : requests) {
        if (request.outcome == nullptr ||
            request.outcome->state == DownloadRangeState::Complete) {
          ::rund::detail::counter::Accumulate(result.confirmed_bytes,
                                              request.bytes);
        }
      }
    }
  } else if (!result.first_failed_valid) {
    for (std::size_t index = 0u; index < requests.size(); ++index) {
      const DownloadRoute &request = requests[index];
      if (request.outcome == nullptr ||
          request.outcome->state == DownloadRangeState::Untouched) {
        MarkDownloadFailure(request.outcome,
                            DownloadRangeState::FailedMayWrite);
        result.first_failed = index;
        result.first_failed_valid = true;
        result.payload_hash_valid = false;
        SetDownloadPrefix(result, requests, index);
        break;
      }
    }
  } else {
    SetDownloadPrefix(
        result, requests,
        std::min<std::size_t>(result.first_failed, requests.size()));
  }
  return result;
}

} // namespace rund::node::accel::detail
