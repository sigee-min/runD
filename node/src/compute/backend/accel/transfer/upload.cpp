#include "../local.hpp"

#include "../../../../accel/context/transfer.hpp"
#include "../../../pipeline/state.hpp"
#include "../../../status.hpp"

namespace rund::compute::detail::accel_backend {

UploadResult upload_pipeline_transfer(PipelineState &pipeline,
                                      const void *const data,
                                      const std::size_t bytes) noexcept {
  const AccelDeviceState *const accel =
      pipeline.device == nullptr ? nullptr : accel_device(*pipeline.device);
  if (accel == nullptr) {
    return UploadResult{.status = Status::fail(Reason::TransferInvalid)};
  }
  const node::accel::detail::AccelTransfer transfer =
      node::accel::detail::UploadPreparedAccelPipeline(
          accel->context, pipeline.prepared, data, bytes);
  return UploadResult{
      .status = transfer.check.ok
                    ? Status::success()
                    : Status::fail(project_reason(transfer.check.reason,
                                                  Reason::TransferInvalid)),
      .staging_bytes = transfer.staging_bytes,
      .staging_peak_bytes = transfer.staging_peak_bytes,
      .staging_reused_bytes = transfer.staging_reused_bytes,
      .staging_budget = transfer.staging_budget,
      .buffer_allocations = transfer.buffer_allocations,
      .buffer_reuses = transfer.buffer_reuses,
      .command_submits = transfer.command_submits,
  };
}

} // namespace rund::compute::detail::accel_backend
