#include "../internal.hpp"
#include "hash.hpp"

#include "../../../../../accel/context/transfer.hpp"

#include <rund/counter.hpp>

#include <array>
#include <limits>

namespace rund::compute::detail::device_vsm_product_detail {
namespace {

using ::rund::detail::counter::Accumulate;

Status write_pages(DeviceVsmProductRun &run, const std::byte *const source,
                   const std::uint64_t bytes) noexcept {
  const std::uint64_t output_pages =
      node::accel::detail::device_vsm_output_page_count(*run.owner->proof);
  const std::uint64_t payload =
      output_pages == 1u ? bytes : run.owner->proof->geometry.payload_bytes;
  Status written = Status::success();
  for (std::uint64_t coordinate = 0u, offset = 0u; offset < bytes;
       ++coordinate, offset += payload) {
    const std::uint64_t count = std::min(payload, bytes - offset);
    written = run.output->write(
        offset,
        std::span<const std::byte>{source + static_cast<std::size_t>(offset),
                                   static_cast<std::size_t>(count)});
    if (!written) {
      run.failed_page = coordinate;
      break;
    }
    Accumulate(run.stats->pipeline.residency.backing_write_bytes, count);
  }
  return written;
}

Status download(DeviceVsmProductRun &run, const AccelDeviceState &native,
                const std::span<std::byte> staged) noexcept {
  std::uint64_t payload_hash = 0u;
  std::array requests{node::accel::detail::DownloadEntry{
      .buffer = &run.owner->output,
      .data = staged.data(),
      .bytes = staged.size(),
      .payload_hash = &payload_hash,
  }};
  std::array<node::accel::detail::DownloadRoute, 1u> routes{};
  const node::accel::detail::AccelTransfer downloaded =
      node::accel::detail::DownloadAccelBuffersMeasured(
          native.context, requests, routes,
          node::accel::detail::TransferAuthority::PipelinePrivate);
  if (!downloaded.check.ok) {
    return Status::fail(
        project_reason(downloaded.check.reason, Reason::TransferInvalid));
  }
  Accumulate(run.stats->buffer_allocations, downloaded.buffer_allocations);
  Accumulate(run.stats->buffer_reuses, downloaded.buffer_reuses);
  Accumulate(run.stats->transfer_submissions.device_to_host,
             downloaded.command_submits);
  Accumulate(run.stats->downloaded_bytes, staged.size());
  Accumulate(run.stats->readback_ns, downloaded.readback_ns);
  return Status::success();
}

[[nodiscard]] Status stage_staged_output(DeviceVsmProductRun &run,
                                         const AccelDeviceState &native,
                                         const std::uint64_t bytes) noexcept {
  if (run.owner == nullptr ||
      run.owner->route_proof.kind() ==
          VirtualDeviceVsmRouteKind::GraphResident ||
      run.owner->route_proof.endpoint() != VirtualDeviceVsmEndpoint::Staged ||
      run.output == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (VirtualBackingAccess::resident(*run.output) != nullptr) {
    return Status::fail(Reason::DeviceLost);
  }
  const node::accel::detail::AccelHostView view =
      node::accel::detail::ReadAccelBuffer(native.context, run.owner->output);
  if (!view || view.bytes < bytes) {
    return Status::fail(Reason::BackendUnsupported);
  }
  run.backing_may_write = true;
  const Status written = write_pages(run, view.data, bytes);
  if (!written) {
    return written;
  }
  run.output_hash = ::rund::node::hash_detail::HashBytes(
      view.data, static_cast<std::size_t>(bytes));
  return Status::success();
}

[[nodiscard]] Status
stage_graph_staged_output(DeviceVsmProductRun &run,
                          const AccelDeviceState &native,
                          const std::uint64_t bytes) noexcept {
  if (run.owner == nullptr ||
      run.owner->route_proof.kind() !=
          VirtualDeviceVsmRouteKind::GraphResident ||
      run.owner->route_proof.endpoint() != VirtualDeviceVsmEndpoint::Staged ||
      run.output == nullptr || run.output_staging.size() != bytes ||
      VirtualBackingAccess::resident(*run.output) != nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  run.backing_may_write = true;
  const Status downloaded = download(run, native, run.output_staging);
  if (!downloaded) {
    return downloaded;
  }
  const Status written = write_pages(run, run.output_staging.data(), bytes);
  if (!written) {
    return written;
  }
  run.output_hash = ::rund::node::hash_detail::HashBytes(
      run.output_staging.data(), static_cast<std::size_t>(bytes));
  return Status::success();
}

} // namespace

Status stage_output(DeviceVsmProductRun &run) noexcept {
  const AccelDeviceState *const native =
      run.state == nullptr || run.state->pipeline == nullptr ||
              run.state->pipeline->device == nullptr
          ? nullptr
          : accel_device(*run.state->pipeline->device);
  if (native == nullptr || run.owner == nullptr || run.output == nullptr ||
      run.projection == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::uint64_t bytes = run.projection->active.output_bytes;
  if (bytes > std::numeric_limits<std::size_t>::max()) {
    return Status::fail(Reason::PipelineCapacity);
  }
  const bool graph_staged =
      run.owner->route_proof.kind() ==
          VirtualDeviceVsmRouteKind::GraphResident &&
      run.owner->route_proof.endpoint() == VirtualDeviceVsmEndpoint::Staged;
  if (graph_staged) {
    const std::uint64_t begin = pipeline_clock();
    const Status status = stage_graph_staged_output(run, *native, bytes);
    run.stats->pipeline.residency.backing_io_ns += pipeline_clock() - begin;
    return status;
  }
  const bool mapped =
      run.owner->route_proof.kind() !=
          VirtualDeviceVsmRouteKind::GraphResident &&
      run.owner->route_proof.endpoint() == VirtualDeviceVsmEndpoint::Staged;
  if (mapped) {
    const std::uint64_t begin = pipeline_clock();
    const Status status = stage_staged_output(run, *native, bytes);
    run.stats->pipeline.residency.backing_io_ns += pipeline_clock() - begin;
    return status;
  }
  // Keep the legacy owner semantics unchanged; StagedLoop sets this only
  // after its mapped output has passed validation above.
  run.backing_may_write = true;
  const bool resident =
      run.owner->resident_output != nullptr &&
      VirtualBackingAccess::resident(*run.output) == run.owner->resident_output;
  if (resident && reuse_output_hash(run)) {
    return Status::success();
  }
  const std::uint64_t begin = pipeline_clock();
  const node::accel::detail::AccelHostView view =
      node::accel::detail::ReadAccelBuffer(native->context, run.owner->output);
  const std::byte *source = view && view.bytes >= bytes ? view.data : nullptr;
  if (resident && source == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (source == nullptr) {
    if (run.output_staging.size() != bytes) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const Status downloaded = download(run, *native, run.output_staging);
    if (!downloaded) {
      return downloaded;
    }
    source = run.output_staging.data();
  }
  const Status written =
      resident ? Status::success() : write_pages(run, source, bytes);
  if (!resident) {
    run.stats->pipeline.residency.backing_io_ns += pipeline_clock() - begin;
  }
  if (!written) {
    return written;
  }
  run.output_hash = ::rund::node::hash_detail::HashBytes(
      source, static_cast<std::size_t>(bytes));
  if (resident) {
    observe_output_hash(run);
  }
  return Status::success();
}

} // namespace rund::compute::detail::device_vsm_product_detail
