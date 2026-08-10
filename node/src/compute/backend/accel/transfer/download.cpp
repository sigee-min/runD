#include "../local.hpp"

#include "../../../../accel/context/transfer.hpp"
#include "../../../pipeline/state.hpp"
#include "../../../status.hpp"

namespace rund::compute::detail::accel_backend {

DownloadResult
download_pipeline_transfer(PipelineState &pipeline, void *const data,
                           const std::size_t bytes,
                           std::uint64_t *const payload_hash) noexcept {
  const AccelDeviceState *const accel =
      pipeline.device == nullptr ? nullptr : accel_device(*pipeline.device);
  if (accel == nullptr) {
    return DownloadResult{.status = Status::fail(Reason::TransferInvalid)};
  }
  const node::accel::detail::AccelTransfer transfer =
      node::accel::detail::DownloadPreparedAccelPipeline(
          accel->context, pipeline.prepared, data, bytes, payload_hash);
  return DownloadResult{
      .status = transfer.check.ok
                    ? Status::success()
                    : Status::fail(project_reason(transfer.check.reason,
                                                  Reason::TransferInvalid)),
      .payload_hash = transfer.payload_hash,
      .staging_bytes = transfer.staging_bytes,
      .staging_peak_bytes = transfer.staging_peak_bytes,
      .staging_reused_bytes = transfer.staging_reused_bytes,
      .staging_budget = transfer.staging_budget,
      .buffer_allocations = transfer.buffer_allocations,
      .buffer_reuses = transfer.buffer_reuses,
      .command_submits = transfer.command_submits,
      .readback_ns = transfer.readback_ns,
      .staging_reused = transfer.staging_reused,
      .payload_hash_valid = transfer.payload_hash_valid,
  };
}

} // namespace rund::compute::detail::accel_backend
