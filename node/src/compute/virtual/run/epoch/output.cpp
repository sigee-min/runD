#include "internal.hpp"

#include "../reduce.hpp"
#include "../scan.hpp"

#include <rund/counter.hpp>

#include <algorithm>
#include <limits>
#include <mutex>

namespace rund::compute::detail {
namespace {

using ::rund::detail::counter::Accumulate;

[[nodiscard]] PipelineFrameDownloadResult
download_frames(PipelineState &pipeline,
                const std::span<const PipelineFrameDownload> frames) noexcept {
  std::lock_guard pipeline_lock{pipeline.gate};
  std::lock_guard publication_lock{pipeline.publication->gate};
  return download_pipeline_private_frames(pipeline, frames);
}

} // namespace

VirtualEpochPhaseResult
finish_virtual_epoch_output(VirtualEpochContext &context) noexcept {
  VirtualEpochPhaseResult result{};
  bool output_poison = false;
  Status status = reserve_residency_output(
      *context.pipeline, context.projected, context.run,
      context.output_reservation, output_poison);
  if (!status) {
    result.status = status;
    result.poison_pipeline = output_poison;
    result.invalidate_all = true;
    return result;
  }
  context.output_lease =
      residency_output_lease(*context.pipeline, context.device_output_lease,
                             context.output_reservation);
  if (context.output_lease.bindings.size() !=
      context.input_lease.bindings.size()) {
    result.status = Status::fail(Reason::PipelineInvalid);
    result.poison_pipeline = true;
    result.invalidate_all = true;
    return result;
  }
  if (context.pipeline->residency_output >=
      context.pipeline->resources.size()) {
    result.status = Status::fail(Reason::PipelineInvalid);
    result.poison_pipeline = true;
    return result;
  }
  const PipelineResource &output_resource =
      context.pipeline->resources[context.pipeline->residency_output];
  for (std::size_t index = 0u; index < context.input_lease.bindings.size();
       ++index) {
    const residency::CacheBinding binding =
        context.device_output_lease.bindings[index];
    std::byte *const output_frame = virtual_resident_output_frame(
        context.run, context.output_lease.bindings[index].frame);
    if (output_frame == nullptr) {
      result.status = Status::fail(Reason::PipelineInvalid);
      result.poison_pipeline = true;
      return result;
    }
    context.download_ranges[index] = PipelineFrameDownload{
        .buffer = output_resource.buffer.get(),
        .data = output_frame,
        .bytes = static_cast<std::size_t>(context.run.output_page_bytes),
        .offset = static_cast<std::size_t>(
            (binding.frame - (context.pool->first_output_frame +
                              context.bank * context.run.frame_capacity)) *
            context.run.output_page_bytes),
        .output = output_resource.output,
    };
  }
  if (context.transaction != nullptr && context.transaction->started &&
      context.transaction->scan) {
    for (const residency::CacheBinding &binding :
         context.output_lease.bindings) {
      const Status journal = record_transaction_output(
          *context.transaction,
          context.pool->authority().virtual_transactions(),
          context.output_lease.token, binding.frame, binding.key,
          residency::FrameTier::Host);
      if (!journal) {
        result.status = journal;
        return result;
      }
    }
  }
  const std::uint64_t expected_download_bytes =
      context.projected.page_count * context.run.output_page_bytes;
  PipelineFrameDownloadResult downloaded =
      context.pipeline->device->backend == Backend::Cpu
          ? PipelineFrameDownloadResult{.bytes = expected_download_bytes}
          : download_frames(*context.pipeline,
                            std::span<const PipelineFrameDownload>{
                                context.download_ranges.data(),
                                context.input_lease.bindings.size()});
  const auto fold_download = [&]() noexcept {
    const Stats physical = pipeline_stats(*context.selected);
    const Status folded = accumulate_virtual_epoch(context.stats, physical);
    if (!folded) {
      return folded;
    }
    return !downloaded.transfer.status ||
                   downloaded.bytes != expected_download_bytes
               ? (downloaded.transfer.status
                      ? Status::fail(Reason::CompletionInvalid)
                      : downloaded.transfer.status)
               : Status::success();
  };
  status = fold_download();
  if (!status) {
    result.status = status;
    result.poison_pipeline = true;
    return result;
  }

  if (context.scan != nullptr) {
    status = prepare_virtual_scan_uniform(context.projected, context.run,
                                          context.input_lease,
                                          context.output_lease, *context.scan);
    if (!status) {
      result.status = status;
      result.failed_page = context.scan->failed_page;
      return result;
    }
    status = upload_virtual_scan_uniform(*context.pipeline, context.run,
                                         context.input_lease, *context.scan,
                                         context.stats, false);
    if (!status) {
      result.status = status;
      result.invalidate_all = true;
      return result;
    }
    status = execute_virtual_epoch_pipeline(context);
    if (!status) {
      const Status restored = upload_virtual_scan_uniform(
          *context.pipeline, context.run, context.input_lease, *context.scan,
          context.stats, true);
      const Stats physical = pipeline_stats(*context.selected);
      const Status folded = accumulate_virtual_epoch(context.stats, physical);
      result.status = !restored ? restored : (folded ? status : folded);
      result.poison_pipeline = poisoned_pipeline(*context.selected);
      result.invalidate_all = !restored;
      if (context.scan->failed_page != ResidencyStats::no_failed_page) {
        result.failed_page = context.scan->failed_page;
      }
      return result;
    }
    downloaded =
        context.pipeline->device->backend == Backend::Cpu
            ? PipelineFrameDownloadResult{.bytes = expected_download_bytes}
            : download_frames(*context.pipeline,
                              std::span<const PipelineFrameDownload>{
                                  context.download_ranges.data(),
                                  context.input_lease.bindings.size()});
    const Status restored = upload_virtual_scan_uniform(
        *context.pipeline, context.run, context.input_lease, *context.scan,
        context.stats, true);
    status = fold_download();
    if (!restored || !status) {
      result.status = !restored ? restored : status;
      result.poison_pipeline = !status;
      result.invalidate_all = !restored;
      return result;
    }
  }

  status = retain_residency_output(*context.pipeline, context.run,
                                   context.device_output_lease, context.stats);
  if (!status) {
    result.status = status;
    return result;
  }

  if (context.reduction != nullptr) {
    status = consume_virtual_reduction(context.run, context.output_lease,
                                       *context.reduction);
    if (!status) {
      result.status = status;
      return result;
    }
  } else if (context.run.reduction) {
    result.status = Status::fail(Reason::PipelineInvalid);
    return result;
  } else {
    for (std::size_t index = 0u; index < context.input_lease.bindings.size();
         ++index) {
      const std::byte *const frame = virtual_resident_output_frame(
          context.run, context.output_lease.bindings[index].frame);
      if (frame == nullptr) {
        result.status = Status::fail(Reason::PipelineInvalid);
        result.poison_pipeline = true;
        return result;
      }
      const std::uint64_t page = context.projected.failed_page + index;
      const std::uint64_t offset = page * context.run.output_payload_bytes;
      const std::size_t bytes = static_cast<std::size_t>(
          std::min(context.run.output_payload_bytes,
                   context.run.active.output_bytes - offset));
      context.output_hash.Bytes(reinterpret_cast<const std::uint8_t *>(
                                    frame + context.run.output_prefix_bytes),
                                bytes);
    }
  }
  context.dirty_count = context.device_output_lease.bindings.size();
  for (std::size_t index = 0u; index < context.dirty_count; ++index) {
    context.dirty[index] = context.device_output_lease.bindings[index].key;
  }
  if (!context.pool->authority().complete(context.execution_token, true)) {
    bool cleaned = context.pool->authority().complete(context.execution_token,
                                                      false, true);
    context.execution_token = 0u;
    cleaned = cancel_residency_output(*context.pipeline,
                                      context.output_reservation) &&
              cleaned;
    cleaned = cancel_virtual_prefetch(context.pool, context.prefetch_pending) &&
              cleaned;
    result.status =
        Status::fail(cleaned ? Reason::PipelineInvalid : Reason::PipelineBusy);
    result.poison_pipeline = true;
    return result;
  }
  context.execution_token = 0u;
  return result;
}

} // namespace rund::compute::detail
