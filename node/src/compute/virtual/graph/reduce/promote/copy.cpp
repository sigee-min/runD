#include "internal.hpp"

#include "../../../../backend.hpp"

#include "../../../../pipeline/run/clock.hpp"
#include "../../../../pipeline/transfer.hpp"

#include <rund/counter.hpp>

namespace rund::compute::detail::graph_reduce::promote_detail {

Status copy(PipelineState &pipeline, const VirtualRunProjection &run,
            const residency::EpochLease destination,
            const Projection &projection, Stats &stats,
            Interval &interval) noexcept {
  if (destination.token == 0u) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (projection.page_count == 0u) {
    interval = {};
    return Status::success();
  }
  UploadProjection upload{};
  if (!project_upload(pipeline, run, projection, upload)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::uint64_t started = pipeline_clock();
  const UploadResult result = pipeline.device->ops->upload_batch(
      *pipeline.device,
      std::span<const UploadRequest>{upload.requests.data(),
                                     upload.request_count},
      node::accel::detail::TransferCompletion::Complete,
      node::accel::detail::TransferAuthority::PipelinePrivate);
  interval = Interval{.started = started, .completed = pipeline_clock()};
  if (!result.status) {
    return result.status;
  }
  record_pipeline_upload(pipeline, upload.bytes, result);
  ::rund::detail::counter::Accumulate(stats.buffer_allocations,
                                      result.buffer_allocations);
  ::rund::detail::counter::Accumulate(stats.buffer_reuses,
                                      result.buffer_reuses);
  ::rund::detail::counter::Accumulate(stats.transfer_submissions.host_to_device,
                                      result.command_submits);
  ::rund::detail::counter::Accumulate(stats.uploaded_bytes, upload.bytes);
  ::rund::detail::counter::Accumulate(stats.host_write_bytes, upload.bytes);
  return Status::success();
}

} // namespace rund::compute::detail::graph_reduce::promote_detail
