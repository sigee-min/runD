#pragma once

#include "../backend.hpp"
#include "state.hpp"

#include <rund/counter.hpp>

#include <algorithm>
#include <cstdint>

namespace rund::compute::detail {

inline void record_pipeline_transfer(PipelineState &state,
                                     const std::uint64_t bytes) noexcept {
  if (bytes == 0u) {
    return;
  }
  ::rund::detail::counter::Accumulate(state.transfer_bytes, bytes);
  state.transfer_peak = std::max(state.transfer_peak, bytes);
}

template <typename Transfer>
inline void record_pipeline_staging(PipelineState &state,
                                    const Transfer &transfer) noexcept {
  state.staging_bytes = ::rund::detail::counter::SaturatingAdd(
      state.staging_bytes, transfer.staging_bytes);
  state.staging_reused = ::rund::detail::counter::SaturatingAdd(
      state.staging_reused, transfer.staging_reused_bytes);
  state.staging_peak =
      std::max(state.staging_peak, transfer.staging_peak_bytes);
  state.staging_budget =
      std::max(state.staging_budget, transfer.staging_budget);
  ::rund::detail::counter::Accumulate(state.stats.buffer_allocations,
                                      transfer.buffer_allocations);
  ::rund::detail::counter::Accumulate(state.stats.buffer_reuses,
                                      transfer.buffer_reuses);
}

inline void record_pipeline_upload(PipelineState &state,
                                   const std::uint64_t bytes,
                                   const UploadResult &transfer) noexcept {
  record_pipeline_staging(state, transfer);
  ::rund::detail::counter::Accumulate(
      state.stats.transfer_submissions.host_to_device,
      transfer.command_submits);
  if (transfer.status && bytes != 0u) {
    ::rund::detail::counter::Accumulate(state.stats.uploaded_bytes, bytes);
    record_pipeline_transfer(state, bytes);
    record_transfer(*state.device, bytes);
  }
}

inline void record_pipeline_download(PipelineState &state,
                                     const std::uint64_t bytes,
                                     const DownloadResult &transfer,
                                     const std::uint64_t events = 1u) noexcept {
  record_pipeline_staging(state, transfer);
  ::rund::detail::counter::Accumulate(
      state.stats.transfer_submissions.device_to_host,
      transfer.command_submits);
  ::rund::detail::counter::Accumulate(state.stats.readback_ns,
                                      transfer.readback_ns);
  if (transfer.status && bytes != 0u) {
    ::rund::detail::counter::Accumulate(state.stats.download_events, events);
    ::rund::detail::counter::Accumulate(state.stats.downloaded_bytes, bytes);
    record_pipeline_transfer(state, bytes);
    record_transfer(*state.device, bytes);
  }
}

} // namespace rund::compute::detail
