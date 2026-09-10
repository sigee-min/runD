#include "internal.hpp"

#include "../../../device/residency/pool.hpp"
#include "../../../pipeline/local.hpp"
#include "../../../pipeline/run/clock.hpp"
#include "../backing.hpp"
#include "../cache.hpp"
#include "../reduce.hpp"
#include "../../stats.hpp"

#include <algorithm>
#include <array>
#include <mutex>
#include <span>

namespace rund::compute::detail::virtual_run_overlap {
namespace {

[[nodiscard]] PipelineFrameDownloadResult
download_frames(PipelineState &pipeline,
                const std::span<const PipelineFrameDownload> frames) noexcept {
  std::lock_guard pipeline_lock{pipeline.gate};
  std::lock_guard publication_lock{pipeline.publication->gate};
  return download_pipeline_private_frames(pipeline, frames);
}

template <bool UseAccelerator>
[[nodiscard]] Status
flush_epoch(PreparedEpoch &prepared, VirtualBacking &output,
            const VirtualRunProjection &run, Stats &stats,
            ::rund::node::hash_detail::Fnv &output_hash,
            VirtualReduction *const reduction, PageOutTimeline &timeline,
            bool &poison) noexcept {
  PipelineState &pipeline = *prepared.pipeline;
  if (pipeline.residency_output >= pipeline.resources.size()) {
    const Status folded = fold_epoch(prepared, stats);
    poison = true;
    return folded ? Status::fail(Reason::PipelineInvalid) : folded;
  }
  const PipelineResource &resource =
      pipeline.resources[pipeline.residency_output];
  const Status reserved =
      prepared.coherent_output
          ? Status::success()
          : reserve_residency_output(pipeline, prepared.projection, run,
                                      prepared.output, poison);
  if (!reserved) {
    const Status folded = fold_epoch(prepared, stats);
    return folded ? reserved : folded;
  }
  const residency::EpochLease output_lease =
      prepared.coherent_output
          ? execution_output_lease(prepared)
          : residency_output_lease(pipeline, execution_output_lease(prepared),
                                    prepared.output);
  if (output_lease.bindings.size() != prepared.binding_count) {
    const Status folded = fold_epoch(prepared, stats);
    poison = !cancel_residency_output(pipeline, prepared.output) || poison;
    return folded ? Status::fail(Reason::PipelineInvalid) : folded;
  }
  std::array<PipelineFrameDownload, PipelineLeafCapacity> downloads{};
  for (std::size_t index = 0u; index < prepared.binding_count; ++index) {
    const std::byte *const target =
        prepared.coherent_output
            ? prepared.output_frames[index]
            : virtual_resident_output_frame(
                  run, output_lease.bindings[index].frame);
    if (target == nullptr) {
      const Status folded = fold_epoch(prepared, stats);
      poison = !cancel_residency_output(pipeline, prepared.output) || poison;
      poison = true;
      return folded ? Status::fail(Reason::PipelineInvalid) : folded;
    }
    downloads[index] = PipelineFrameDownload{
        .buffer = resource.buffer.get(),
        .data = const_cast<std::byte *>(target),
        .bytes = static_cast<std::size_t>(run.output_page_bytes),
        .offset = static_cast<std::size_t>(
            (prepared.bindings[prepared.binding_count + index].frame -
             (pipeline.residency_pool->first_output_frame +
              pipeline.residency_bank * run.frame_capacity)) *
            run.output_page_bytes),
        .output = resource.output,
    };
  }
  const std::uint64_t expected =
      prepared.projection.page_count * run.output_page_bytes;
  PipelineFrameDownloadResult downloaded{};
  if constexpr (!UseAccelerator) {
    downloaded.bytes = expected;
  } else if (prepared.coherent_output) {
    downloaded.bytes = expected;
  } else {
    timeline.download.started = pipeline_clock();
    downloaded = download_frames(
        pipeline, std::span<const PipelineFrameDownload>{
                      downloads.data(), prepared.binding_count});
    timeline.download.completed = pipeline_clock();
  }
  const Status folded = fold_epoch(prepared, stats);
  if (!folded || !downloaded.transfer.status || downloaded.bytes != expected) {
    bool discarded = cancel_residency_output(pipeline, prepared.output);
    if (prepared.drain_token != 0u) {
      discarded =
          pipeline.residency_pool->authority().discard(prepared.drain_token) &&
          discarded;
      prepared.drain_token = 0u;
    }
    poison = !folded || !discarded || poison;
    return !folded
               ? folded
               : (downloaded.transfer.status
                      ? Status::fail(Reason::CompletionInvalid)
                      : downloaded.transfer.status);
  }
  residency::Pool &pool = *pipeline.residency_pool;
  if constexpr (UseAccelerator) {
    if (!prepared.coherent_output) {
      const std::uint64_t device_token = prepared.drain_token;
      const std::uint64_t output_token = prepared.output.token;
      const bool migrated =
          pool.authority().complete_migration(device_token, output_token);
      if (!migrated) {
        const bool device_terminal =
            device_token != 0u && pool.authority().discard(device_token);
        const bool output_terminal =
            cancel_residency_output(pipeline, prepared.output);
        prepared.drain_token = 0u;
        poison = !device_terminal || !output_terminal || poison;
        return Status::fail(Reason::PipelineInvalid);
      }
      prepared.drain_token = 0u;
      prepared.output.token = 0u;
      std::array<residency::CacheKey, PipelineLeafCapacity> output_keys{};
      for (std::size_t index = 0u; index < output_lease.bindings.size();
           ++index) {
        output_keys[index] = output_lease.bindings[index].key;
      }
      const std::span<const residency::CacheKey> host_keys{
          output_keys.data(), output_lease.bindings.size()};
      const auto begin_host_drain = [&]() noexcept {
        return run.reduction
                   ? pool.authority().begin_discard(
                         host_keys, pool.first_host_output_frame,
                         pool.host_output_frame_count)
                   : pool.authority().begin_writeback(
                         host_keys, pool.first_host_output_frame,
                         pool.host_output_frame_count);
      };
      const residency::AuthorityResult writeback = begin_host_drain();
      if (!writeback ||
          writeback.lease.transitions.size() > prepared.drains.size()) {
        bool clean = true;
        if (writeback) {
          clean = pool.authority().discard(writeback.lease.token);
        } else {
          const residency::AuthorityResult recovery = begin_host_drain();
          clean = recovery && pool.authority().discard(recovery.lease.token);
        }
        poison = !clean || poison;
        return Status::fail(clean ? Reason::PipelineInvalid
                                  : Reason::PipelineBusy);
      }
      prepared.drain_token = writeback.lease.token;
      prepared.drain_count = writeback.lease.transitions.size();
      std::copy(writeback.lease.transitions.begin(),
                writeback.lease.transitions.end(), prepared.drains.begin());
    }
  }
  if (reduction != nullptr) {
    const Status consumed =
        consume_virtual_reduction(run, output_lease, *reduction);
    if (!consumed) {
      return consumed;
    }
  } else if (run.reduction) {
    return Status::fail(Reason::PipelineInvalid);
  } else {
    for (std::size_t index = 0u; index < prepared.binding_count; ++index) {
      const std::uint64_t page = prepared.projection.failed_page + index;
      const std::uint64_t logical = page * run.output_payload_bytes;
      const std::size_t bytes = static_cast<std::size_t>(std::min(
          run.output_payload_bytes, run.active.output_bytes - logical));
      const std::byte *const frame =
          prepared.coherent_output
              ? prepared.output_frames[index]
              : virtual_resident_output_frame(
                    run, output_lease.bindings[index].frame);
      if (frame == nullptr) {
        poison = true;
        return Status::fail(Reason::PipelineInvalid);
      }
      output_hash.Bytes(reinterpret_cast<const std::uint8_t *>(
                            frame + run.output_prefix_bytes),
                        bytes);
    }
  }
  if (prepared.drain_token != 0u && reduction != nullptr) {
    const bool discarded = pool.authority().discard(prepared.drain_token);
    prepared.drain_token = 0u;
    if (!discarded) {
      poison = true;
      return Status::fail(Reason::PipelineBusy);
    }
  } else if (prepared.drain_token != 0u) {
    VirtualTransferInterval backing_interval{};
    const Status written = writeback_residency_cache(
        output, run,
        std::span<const residency::CacheTransition>{prepared.drains.data(),
                                                    prepared.drain_count},
        stats.pipeline.residency, &backing_interval,
        prepared.coherent_output
            ? std::span<const std::byte *const>{prepared.output_frames.data(),
                                                prepared.drain_count}
            : std::span<const std::byte *const>{});
    timeline.backing = TimelineInterval{
        .started = backing_interval.started_ns,
        .completed = backing_interval.completed_ns,
    };
    const bool published =
        written && pipeline.residency_pool->authority().complete(
                       prepared.drain_token, true);
    bool completed = written
                        ? published
                        : pipeline.residency_pool->authority().discard(
                              prepared.drain_token);
    if (written && !published) {
      completed =
          pipeline.residency_pool->authority().discard(prepared.drain_token);
    }
    prepared.drain_token = 0u;
    if (!written || !published || !completed) {
      poison = (written && !published) || !completed || poison;
      return written ? Status::fail(Reason::PipelineInvalid) : written;
    }
  }
  return Status::success();
}

} // namespace

Status flush_cpu_epoch(
    PreparedEpoch &prepared, VirtualBacking &output,
    const VirtualRunProjection &run, Stats &stats,
    ::rund::node::hash_detail::Fnv &output_hash,
    VirtualReduction *const reduction, PageOutTimeline &timeline,
    bool &poison) noexcept {
  return flush_epoch<false>(prepared, output, run, stats, output_hash, reduction,
                            timeline, poison);
}

Status flush_accel_epoch(
    PreparedEpoch &prepared, VirtualBacking &output,
    const VirtualRunProjection &run, Stats &stats,
    ::rund::node::hash_detail::Fnv &output_hash,
    VirtualReduction *const reduction, PageOutTimeline &timeline,
    bool &poison) noexcept {
  return flush_epoch<true>(prepared, output, run, stats, output_hash, reduction,
                           timeline, poison);
}

} // namespace rund::compute::detail::virtual_run_overlap
