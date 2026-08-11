#include "epoch.hpp"

#include "../../device/residency_pool.hpp"
#include "../../pipeline/local.hpp"
#include "../../pipeline/run/clock.hpp"
#include "../backing.hpp"
#include "../stats.hpp"
#include "backing.hpp"
#include "cache.hpp"
#include "reduce.hpp"
#include "scan.hpp"

#include <rund/counter.hpp>

#include <array>
#include <limits>
#include <mutex>
#include <span>

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

VirtualEpochResult execute_virtual_epoch(
    VirtualPipelineState &state, VirtualBacking &input, VirtualBacking &output,
    const VirtualRunProjection &run, const std::uint64_t epoch,
    std::array<bool, 2u> &prefetch_pending, Stats &stats,
    ::rund::node::hash_detail::Fnv &output_hash,
    VirtualReduction *const reduction, VirtualScan *const scan) noexcept {
  VirtualEpochProjection projected{};
  if (!project_virtual_epoch(run, epoch, projected)) {
    return VirtualEpochResult{.status = Status::fail(Reason::PipelineInvalid),
                              .failed_page = projected.failed_page,
                              .poison_pipeline = true};
  }

  PipelineState &pipeline = *state.pipeline;
  if (pipeline.residency_pool == nullptr ||
      projected.page_count > PipelineLeafCapacity) {
    return VirtualEpochResult{.status = Status::fail(Reason::PipelineInvalid),
                              .failed_page = projected.failed_page,
                              .poison_pipeline = true};
  }
  residency::Pool &pool = *pipeline.residency_pool;
  std::array<residency::CacheUse, PipelineLeafCapacity> cache_uses{};
  const std::uint64_t backing_id = VirtualBackingAccess::id(input);
  const std::uint64_t backing_version = VirtualBackingAccess::version(input);
  for (std::size_t index = 0u;
       index < static_cast<std::size_t>(projected.page_count); ++index) {
    cache_uses[index] = residency::CacheUse{
        .key =
            residency::CacheKey{
                .backing = backing_id,
                .version = backing_version,
                .extent = run.cache_extent,
                .materialization_hi = run.cache_identity_hi,
                .materialization_lo = run.cache_identity_lo,
                .page = projected.failed_page + index,
            },
        .access = run.reduction ? residency::Access::Read
                                : residency::Access::ReadWrite,
        .next_use = std::numeric_limits<std::uint64_t>::max(),
    };
  }
  const residency::AuthorityResult acquired =
      pipeline.residency_pool->authority.begin(
          std::span<const residency::CacheUse>{
              cache_uses.data(),
              static_cast<std::size_t>(projected.page_count)});
  if (!acquired) {
    return VirtualEpochResult{
        .status =
            Status::fail(acquired.failure == residency::AuthorityFailure::Busy
                             ? Reason::PipelineBusy
                             : Reason::PipelineMemoryBudget),
        .failed_page = projected.failed_page};
  }
  const auto abort = [&](const Status status,
                         const bool poison = false) noexcept {
    for (std::size_t lane = 0u; lane < prefetch_pending.size(); ++lane) {
      if (prefetch_pending[lane]) {
        (void)pool.prefetch[lane].wait();
        prefetch_pending[lane] = false;
      }
    }
    const bool rolled_back = pipeline.residency_pool->authority.complete(
        acquired.lease.token, false);
    return VirtualEpochResult{
        .status = rolled_back ? status : Status::fail(Reason::PipelineInvalid),
        .failed_page = projected.failed_page,
        .poison_pipeline = poison || !rolled_back,
    };
  };

  Status status = writeback_residency_cache(
      output, run, acquired.lease.transitions, stats.pipeline.residency);
  if (!status) {
    return abort(status);
  }

  const std::size_t consume_lane = static_cast<std::size_t>(epoch % 2u);
  const residency::PrefetchReceipt prefetched =
      prefetch_pending[consume_lane] ? pool.prefetch[consume_lane].wait()
                                     : residency::PrefetchReceipt{};
  prefetch_pending[consume_lane] = false;

  const std::uint64_t supply_started = pipeline_clock();
  const VirtualSupplyResult supplied =
      read_virtual_epoch(input, projected, run, acquired.lease, prefetched,
                         stats.pipeline.residency);
  if (!supplied.status) {
    return abort(supplied.status);
  }
  // The backing callback is the byte-traffic authority. Once it succeeds,
  // those bytes were read even if a later resident upload is rejected.
  std::uint64_t fetched_pages = 0u;
  std::uint64_t cache_hits = 0u;
  for (std::size_t index = 0u; index < acquired.lease.bindings.size();
       ++index) {
    if (!acquired.lease.bindings[index].fetch) {
      ++cache_hits;
      continue;
    }
    ++fetched_pages;
  }
  std::uint64_t evictions = 0u;
  for (const residency::CacheTransition transition :
       acquired.lease.transitions) {
    evictions += static_cast<std::uint64_t>(transition.kind ==
                                            residency::TransitionKind::Unmap);
  }
  Accumulate(stats.pipeline.residency.cache_hit_count, cache_hits);
  Accumulate(stats.pipeline.residency.eviction_count, evictions);
  if (supplied.fetched_pages != fetched_pages) {
    return abort(Status::fail(Reason::PipelineInvalid));
  }
  Accumulate(stats.pipeline.residency.late_page_count, supplied.late_pages);
  Accumulate(stats.pipeline.residency.page_in_bytes, supplied.backing_bytes);

  if (scan != nullptr) {
    const Status prepared_scan =
        prepare_virtual_scan_page(run, acquired.lease, *scan);
    if (!prepared_scan) {
      return abort(prepared_scan);
    }
  }

  status = supply_residency_cache(pipeline, run, acquired.lease,
                                  scan != nullptr, stats);
  if (!status) {
    return abort(status);
  }
  Accumulate(stats.pipeline.residency.stall_ns,
             pipeline_clock() - supply_started);
  Accumulate(stats.pipeline.residency.page_in_count, fetched_pages);

  const std::uint64_t lookahead =
      input.tier() == VirtualBackingTier::Persistent &&
              input.max_parallel_reads() >= 2u
          ? 2u
          : 1u;
  for (std::uint64_t distance = 1u; distance <= lookahead; ++distance) {
    if (epoch > std::numeric_limits<std::uint64_t>::max() - distance ||
        epoch + distance >= run.active.stream.epoch_count()) {
      break;
    }
    const std::size_t lane = static_cast<std::size_t>((epoch + distance) % 2u);
    if (prefetch_pending[lane]) {
      continue;
    }
    VirtualEpochProjection next{};
    if (!project_virtual_epoch(run, epoch + distance, next)) {
      return abort(Status::fail(Reason::PipelineInvalid));
    }
    status =
        schedule_virtual_prefetch(input, next, run, pool.authority,
                                  pool.prefetch[lane], prefetch_pending[lane]);
    if (!status) {
      return abort(status);
    }
  }

  status = run_pipeline_with_mode(
      state.pipeline, node::accel::detail::PipelineSubmitMode::Residency);
  if (!status) {
    const Stats wave_stats = pipeline_stats(state.pipeline);
    const Status folded = accumulate_virtual_epoch(stats, wave_stats);
    return abort(folded ? status : folded, poisoned_pipeline(state.pipeline));
  }
  Accumulate(stats.pipeline.residency.epoch_count, 1u);

  std::array<PipelineFrameDownload, PipelineLeafCapacity> download_ranges{};
  for (std::size_t index = 0u; index < acquired.lease.bindings.size();
       ++index) {
    const residency::CacheBinding binding = acquired.lease.bindings[index];
    download_ranges[index] = PipelineFrameDownload{
        .buffer = run.download.buffer,
        .data = run.output_stage + binding.frame * run.output_page_bytes,
        .bytes = static_cast<std::size_t>(run.output_page_bytes),
        .offset =
            static_cast<std::size_t>(binding.frame * run.output_page_bytes),
        .output = run.download.output,
    };
  }
  const PipelineFrameDownloadResult downloaded = download_frames(
      pipeline, std::span<const PipelineFrameDownload>{
                    download_ranges.data(), acquired.lease.bindings.size()});
  const Stats wave_stats = pipeline_stats(state.pipeline);
  const Status folded = accumulate_virtual_epoch(stats, wave_stats);
  if (!folded) {
    return abort(folded, true);
  }
  const std::uint64_t expected_download_bytes =
      projected.page_count * run.output_page_bytes;
  if (!downloaded.transfer.status ||
      downloaded.bytes != expected_download_bytes) {
    return abort(downloaded.transfer.status
                     ? Status::fail(Reason::CompletionInvalid)
                     : downloaded.transfer.status);
  }

  status = retain_residency_output(pipeline, run, acquired.lease, stats);
  if (!status) {
    return abort(status);
  }

  if (scan != nullptr) {
    status = complete_virtual_scan_page(input, projected, run, acquired.lease,
                                        *scan, stats.pipeline.residency);
    if (!status) {
      return abort(status);
    }
  }

  if (reduction != nullptr) {
    status = consume_virtual_reduction(run, acquired.lease, *reduction);
    if (!status) {
      return abort(status);
    }
  } else if (run.reduction) {
    return abort(Status::fail(Reason::PipelineInvalid));
  } else {
    for (std::size_t index = 0u; index < acquired.lease.bindings.size();
         ++index) {
      const residency::CacheBinding binding = acquired.lease.bindings[index];
      const std::uint64_t page = projected.failed_page + index;
      const std::uint64_t offset = page * run.output_payload_bytes;
      const std::size_t bytes = static_cast<std::size_t>(
          std::min(run.output_payload_bytes, run.active.output_bytes - offset));
      output_hash.Bytes(reinterpret_cast<const std::uint8_t *>(
                            run.output_stage +
                            binding.frame * run.output_page_bytes +
                            run.output_prefix_bytes),
                        bytes);
    }
  }
  if (!pipeline.residency_pool->authority.complete(acquired.lease.token,
                                                   true)) {
    return VirtualEpochResult{.status = Status::fail(Reason::PipelineInvalid),
                              .failed_page = projected.failed_page,
                              .poison_pipeline = true};
  }
  return {};
}

} // namespace rund::compute::detail
