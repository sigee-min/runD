#pragma once

#include <node/accel/context.hpp>

#include "local.hpp"

namespace node_accel_contract::context::reject {

[[nodiscard]] inline bool RejectsTransferRangeOverflow(
    const State& state,
    TransferFixture& io) {
  if (!CheckReason(rund::node::accel::UploadAccelBuffer(
                       state.context, state.created, io.upload.data(),
                       state.created.byte_extent + 1u),
                   "accel_buffer_upload_overflow") ||
      !CheckReason(rund::node::accel::DownloadAccelBuffer(
                       state.context, state.created, io.download.data(),
                       state.created.byte_extent + 1u),
                   "accel_buffer_download_overflow")) {
    return false;
  }

  constexpr std::uint64_t kMax = std::numeric_limits<std::uint64_t>::max();
  return CheckReason(rund::node::accel::UploadAccelBuffer(
                         state.context, state.created, io.upload.data(), 1u,
                         kMax),
                     "accel_buffer_upload_overflow") &&
         CheckReason(rund::node::accel::DownloadAccelBuffer(
                         state.context, state.created, io.download.data(), 1u,
                         kMax),
                     "accel_buffer_download_overflow");
}

}  // namespace node_accel_contract::context::reject
