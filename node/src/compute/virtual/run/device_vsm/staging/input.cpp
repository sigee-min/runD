#include "../internal.hpp"
#include "cohort.hpp"

#include "../../../../../accel/context/transfer.hpp"

#include <rund/counter.hpp>

#include <algorithm>
#include <array>
#include <limits>

namespace rund::compute::detail::device_vsm_product_detail {
namespace {

using ::rund::detail::counter::Accumulate;

[[nodiscard]] bool output_ok(VirtualBacking *const output,
                             const std::uint64_t id,
                             const std::uint64_t version,
                             const std::uint64_t bytes) noexcept {
  return output != nullptr &&
         VirtualBackingAccess::resident(*output) == nullptr &&
         output->size_bytes() == bytes &&
         VirtualBackingAccess::id(*output) == id &&
         VirtualBackingAccess::version(*output) == version;
}

[[nodiscard]] bool
cohort_reads_ok(const std::span<VirtualBacking *const> inputs,
                const std::span<const VirtualCohortRead> reads,
                const std::shared_ptr<VirtualReadCohort> &provider,
                const VirtualCohortId cohort, const std::uint32_t lanes,
                const std::uint64_t bytes, const std::uint64_t page_bytes,
                const std::uint64_t page_count) noexcept {
  if (provider == nullptr || inputs.empty() || inputs.size() != reads.size() ||
      inputs.size() > VirtualPipelineState::InputCapacity || page_bytes == 0u ||
      page_count == 0u ||
      page_count > std::numeric_limits<std::uint64_t>::max() / page_bytes ||
      bytes > page_bytes * page_count || provider->cohort_id() != cohort ||
      provider->lane_limit() != lanes) {
    return false;
  }
  const std::shared_ptr<VirtualReadCohort> current = graph_read_cohort(inputs);
  if (current == nullptr || current.get() != provider.get()) {
    return false;
  }
  for (std::size_t index = 0u; index < inputs.size(); ++index) {
    VirtualBacking *const backing = inputs[index];
    const VirtualCohortRead &read = reads[index];
    if (backing == nullptr || read.backing != backing ||
        backing->size_bytes() != bytes ||
        read.backing_id != VirtualBackingAccess::id(*backing) ||
        read.version != VirtualBackingAccess::version(*backing) ||
        read.page_bytes != page_bytes || read.page_count != page_count ||
        read.destination.data() == nullptr ||
        read.destination.size() != bytes) {
      return false;
    }
  }
  return true;
}

Status read_pages(DeviceVsmProductRun &run, VirtualBacking &backing,
                  std::byte *const target, const std::uint64_t bytes) noexcept {
  const std::uint64_t payload = run.owner->proof->geometry.payload_bytes;
  Status read = Status::success();
  for (std::uint64_t coordinate = 0u, offset = 0u; offset < bytes;
       ++coordinate, offset += payload) {
    const std::uint64_t count = std::min(payload, bytes - offset);
    read = backing.read(
        offset, std::span<std::byte>{target + static_cast<std::size_t>(offset),
                                     static_cast<std::size_t>(count)});
    if (!read) {
      run.failed_page = coordinate;
      break;
    }
    Accumulate(run.stats->pipeline.residency.backing_read_bytes, count);
  }
  return read;
}

Status upload(
    DeviceVsmProductRun &run, const AccelDeviceState &native,
    const std::span<const node::accel::detail::UploadEntry> requests) noexcept {
  std::array<node::accel::detail::UploadRoute,
             VirtualPipelineState::InputCapacity>
      routes{};
  const node::accel::detail::AccelTransfer uploaded =
      node::accel::detail::UploadAccelBuffers(
          native.context, requests,
          std::span<node::accel::detail::UploadRoute>{routes.data(),
                                                      requests.size()},
          node::accel::detail::TransferCompletion::Complete,
          node::accel::detail::TransferAuthority::PipelinePrivate);
  if (!uploaded.check.ok) {
    return Status::fail(
        project_reason(uploaded.check.reason, Reason::TransferInvalid));
  }
  Accumulate(run.stats->buffer_allocations, uploaded.buffer_allocations);
  Accumulate(run.stats->buffer_reuses, uploaded.buffer_reuses);
  Accumulate(run.stats->transfer_submissions.host_to_device,
             uploaded.command_submits);
  for (const node::accel::detail::UploadEntry &request : requests) {
    Accumulate(run.stats->uploaded_bytes, request.bytes);
    Accumulate(run.stats->host_write_bytes, request.bytes);
  }
  return Status::success();
}

[[nodiscard]] Status stage_staged_input(DeviceVsmProductRun &run,
                                        const AccelDeviceState &native,
                                        const std::uint64_t bytes) noexcept {
  if (run.owner == nullptr ||
      run.owner->route_proof.kind == VirtualDeviceVsmRouteKind::GraphResident ||
      run.owner->route_proof.endpoint != VirtualDeviceVsmEndpoint::Staged ||
      run.input_count == 0u) {
    return Status::fail(Reason::PipelineInvalid);
  }
  for (std::size_t index = 0u; index < run.input_count; ++index) {
    VirtualBacking *const backing = run.inputs[index];
    if (backing == nullptr || run.owner->resident_inputs[index] != nullptr) {
      return Status::fail(Reason::PipelineInvalid);
    }
    if (VirtualBackingAccess::resident(*backing) != nullptr) {
      return Status::fail(Reason::DeviceLost);
    }
    const node::accel::detail::AccelHostWriteView view =
        node::accel::detail::WriteAccelBuffer(native.context,
                                              run.owner->inputs[index]);
    if (!view || view.bytes < bytes) {
      return Status::fail(Reason::BackendUnsupported);
    }
    const Status read = read_pages(run, *backing, view.data, bytes);
    if (!read) {
      return read;
    }
    Accumulate(run.stats->host_write_bytes, bytes);
  }
  return Status::success();
}

[[nodiscard]] Status stage_cohort(DeviceVsmProductRun &run,
                                  const AccelDeviceState &native,
                                  const std::uint64_t bytes) noexcept {
  if (run.projection == nullptr || run.owner == nullptr ||
      run.owner->proof == nullptr ||
      run.owner->route_proof.kind != VirtualDeviceVsmRouteKind::GraphResident ||
      run.owner->route_proof.endpoint != VirtualDeviceVsmEndpoint::Staged ||
      run.owner->proof->topology !=
          node::accel::detail::DeviceVsmTopology::GraphResident ||
      run.output == nullptr ||
      run.projection->active.graph.page_count() == 0u) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::span<VirtualBacking *const> inputs{run.inputs.data(),
                                                run.input_count};
  const std::shared_ptr<VirtualReadCohort> provider = graph_read_cohort(inputs);
  if (provider == nullptr) {
    return Status::fail(Reason::DeviceLost);
  }
  const VirtualCohortId cohort = provider->cohort_id();
  const std::uint32_t lanes = provider->lane_limit();
  const std::uint64_t output_id = VirtualBackingAccess::id(*run.output);
  const std::uint64_t output_version =
      VirtualBackingAccess::version(*run.output);
  if (!output_ok(run.output, output_id, output_version, bytes)) {
    return Status::fail(Reason::DeviceLost);
  }
  if (run.input_count == 0u ||
      bytes > std::numeric_limits<std::uint64_t>::max() / run.input_count) {
    return Status::fail(Reason::PipelineCapacity);
  }
  const std::uint64_t expected = bytes * run.input_count;
  std::array<VirtualCohortRead, VirtualPipelineState::InputCapacity> reads{};
  std::array<node::accel::detail::UploadEntry,
             VirtualPipelineState::InputCapacity>
      requests{};
  std::size_t request_count = 0u;
  for (std::size_t index = 0u; index < run.input_count; ++index) {
    VirtualBacking *const backing = run.inputs[index];
    if (backing == nullptr) {
      return Status::fail(Reason::DeviceLost);
    }
    const std::span<std::byte> destination = run.input_staging[index];
    if (destination.size() != bytes || request_count >= requests.size()) {
      return Status::fail(Reason::PipelineInvalid);
    }
    requests[request_count++] = node::accel::detail::UploadEntry{
        .buffer = &run.owner->inputs[index],
        .data = destination.data(),
        .bytes = destination.size(),
    };
    reads[index] = VirtualCohortRead{
        .backing = backing,
        .backing_id = VirtualBackingAccess::id(*backing),
        .version = VirtualBackingAccess::version(*backing),
        .page_bytes = run.projection->input_page_bytes,
        .page_count = run.projection->active.graph.page_count(),
        .destination = destination,
    };
  }
  const std::span<const VirtualCohortRead> read_span{reads.data(),
                                                     run.input_count};
  if (!cohort_reads_ok(inputs, read_span, provider, cohort, lanes, bytes,
                       run.projection->input_page_bytes,
                       run.projection->active.graph.page_count())) {
    return Status::fail(Reason::DeviceLost);
  }
  VirtualCohortResult result{};
  Status status = provider->read_cohort(
      std::span<VirtualCohortRead>{reads.data(), run.input_count}, result);
  if (!cohort_reads_ok(inputs, read_span, provider, cohort, lanes, bytes,
                       run.projection->input_page_bytes,
                       run.projection->active.graph.page_count())) {
    return Status::fail(Reason::DeviceLost);
  }
  if (!output_ok(run.output, output_id, output_version, bytes)) {
    return Status::fail(Reason::DeviceLost);
  }
  const std::uint64_t no_index = std::numeric_limits<std::uint64_t>::max();
  if (!status) {
    if (!result.joined || result.failed_member >= run.input_count ||
        result.failed_page >= run.projection->active.graph.page_count() ||
        result.completed_bytes > expected) {
      return Status::fail(Reason::DeviceLost);
    }
    Accumulate(run.stats->pipeline.residency.backing_read_bytes,
               result.completed_bytes);
    run.failed_input = result.failed_member;
    run.failed_page = result.failed_page;
    return status;
  }
  if (!result.joined || result.failed_member != no_index ||
      result.failed_page != no_index || result.completed_bytes != expected) {
    return Status::fail(Reason::DeviceLost);
  }
  Accumulate(run.stats->pipeline.residency.backing_read_bytes,
             result.completed_bytes);
  if (request_count != 0u) {
    status = upload(run, native,
                    std::span<const node::accel::detail::UploadEntry>{
                        requests.data(), request_count});
  }
  return status;
}

} // namespace

Status stage_input(DeviceVsmProductRun &run) noexcept {
  const AccelDeviceState *const native =
      run.state == nullptr || run.state->pipeline == nullptr ||
              run.state->pipeline->device == nullptr
          ? nullptr
          : accel_device(*run.state->pipeline->device);
  if (native == nullptr || run.owner == nullptr || run.projection == nullptr ||
      run.input_count == 0u || run.input_count != run.owner->input_count ||
      run.input_count > run.inputs.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::uint64_t bytes = run.projection->active.input_bytes;
  if (bytes > std::numeric_limits<std::size_t>::max()) {
    return Status::fail(Reason::PipelineCapacity);
  }
  const std::uint64_t begin = pipeline_clock();
  const bool graph_staged =
      run.input_count > 1u && run.projection->graph_execution &&
      run.owner->proof != nullptr &&
      run.owner->route_proof.kind == VirtualDeviceVsmRouteKind::GraphResident &&
      run.owner->route_proof.endpoint == VirtualDeviceVsmEndpoint::Staged &&
      run.owner->proof->topology ==
          node::accel::detail::DeviceVsmTopology::GraphResident &&
      std::all_of(run.owner->resident_inputs.begin(),
                  run.owner->resident_inputs.begin() + run.input_count,
                  [](const std::shared_ptr<BufferState> &resident) {
                    return resident == nullptr;
                  });
  if (graph_staged) {
    const Status status = stage_cohort(run, *native, bytes);
    run.stats->pipeline.residency.backing_io_ns += pipeline_clock() - begin;
    return status;
  }
  const bool mapped =
      run.owner->route_proof.kind != VirtualDeviceVsmRouteKind::GraphResident &&
      run.owner->route_proof.endpoint == VirtualDeviceVsmEndpoint::Staged;
  if (mapped) {
    const Status status = stage_staged_input(run, *native, bytes);
    run.stats->pipeline.residency.backing_io_ns += pipeline_clock() - begin;
    return status;
  }
  std::array<node::accel::detail::UploadEntry,
             VirtualPipelineState::InputCapacity>
      requests{};
  std::size_t request_count = 0u;
  Status read = Status::success();
  bool serviced = false;
  for (std::size_t index = 0u; read && index < run.input_count; ++index) {
    VirtualBacking *const backing = run.inputs[index];
    if (backing == nullptr) {
      read = Status::fail(Reason::PipelineInvalid);
      break;
    }
    if (run.owner->resident_inputs[index] != nullptr &&
        VirtualBackingAccess::resident(*backing) ==
            run.owner->resident_inputs[index]) {
      continue;
    }
    serviced = true;
    // These are privately admitted whole-run buffers. Prefer their
    // authenticated coherent view independently of the route label; the
    // allocation may also be mapped for legacy Direct Graph collectives.
    // Only the absence of that view requires a native upload submission.
    const node::accel::detail::AccelHostWriteView view =
        node::accel::detail::WriteAccelBuffer(native->context,
                                              run.owner->inputs[index]);
    if (view && view.bytes >= bytes) {
      read = read_pages(run, *backing, view.data, bytes);
      if (read) {
        Accumulate(run.stats->host_write_bytes, bytes);
      }
      continue;
    }
    const std::span<std::byte> staged = run.input_staging[index];
    if (staged.size() != bytes) {
      read = Status::fail(Reason::PipelineInvalid);
      break;
    }
    read = read_pages(run, *backing, staged.data(), bytes);
    if (read) {
      requests[request_count++] = node::accel::detail::UploadEntry{
          .buffer = &run.owner->inputs[index],
          .data = staged.data(),
          .bytes = staged.size(),
      };
    }
  }
  if (read && request_count != 0u) {
    read = upload(run, *native,
                  std::span<const node::accel::detail::UploadEntry>{
                      requests.data(), request_count});
  }
  if (serviced) {
    run.stats->pipeline.residency.backing_io_ns += pipeline_clock() - begin;
  }
  return read;
}

} // namespace rund::compute::detail::device_vsm_product_detail
