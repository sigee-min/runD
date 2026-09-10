#include "internal.hpp"

#include "../backing.hpp"
#include "../scan.hpp"

#include <rund/counter.hpp>

#include <limits>

namespace rund::compute::detail {

using ::rund::detail::counter::Accumulate;

VirtualEpochPhaseResult
supply_virtual_epoch(VirtualEpochContext &context) noexcept {
  VirtualEpochPhaseResult result{};
  Status status = writeback_residency_cache(
      context.output, context.run, context.acquired.lease.transitions,
      context.stats.pipeline.residency, nullptr, {}, context.transaction);
  if (!status) {
    result.status = status;
    return result;
  }

  const std::uint64_t supply_started = pipeline_clock();
  const VirtualSupplyResult supplied = read_virtual_epoch(
      context.input, context.projected, context.run, context.input_lease,
      context.prefetched, context.stats.pipeline.residency);
  if (!supplied.status) {
    result.status = supplied.status;
    return result;
  }
  // The backing callback is the byte-traffic authority. Once it succeeds,
  // those bytes were read even if a later resident upload is rejected.
  std::uint64_t fetched_pages = 0u;
  std::uint64_t cache_hits = 0u;
  for (std::size_t index = 0u; index < context.input_lease.bindings.size();
       ++index) {
    if (!context.input_lease.bindings[index].fetch) {
      ++cache_hits;
      continue;
    }
    ++fetched_pages;
  }
  std::uint64_t evictions = 0u;
  const residency::FrameRegion input_region =
      context.pool->input_regions[context.bank];
  const std::uint64_t input_bank_first = input_region.first;
  for (const residency::CacheTransition &transition :
       context.acquired.lease.transitions) {
    evictions += static_cast<std::uint64_t>(
        transition.kind == residency::TransitionKind::Unmap &&
        transition.frame >= input_bank_first &&
        transition.frame - input_bank_first < context.run.frame_capacity);
  }
  Accumulate(context.stats.pipeline.residency.cache_hit_count, cache_hits);
  Accumulate(context.stats.pipeline.residency.eviction_count, evictions);
  if (supplied.fetched_pages != fetched_pages) {
    result.status = Status::fail(Reason::PipelineInvalid);
    return result;
  }
  Accumulate(context.stats.pipeline.residency.late_page_count,
             supplied.late_pages);
  Accumulate(context.stats.pipeline.residency.page_in_bytes,
             context.pool->device->backend == Backend::Cpu
                 ? supplied.backing_bytes
                 : fetched_pages * context.run.input_page_bytes);

  if (context.scan != nullptr) {
    status = capture_virtual_scan_input(
        context.projected, context.run, context.input_lease,
        context.pool->device->backend == Backend::Cpu ? nullptr
                                                      : &context.prefetched,
        *context.scan);
    if (!status) {
      result.status = status;
      return result;
    }
  }

  status = supply_residency_cache(
      *context.pipeline, context.run, context.input_lease, context.stats,
      context.pool->device->backend == Backend::Cpu ? nullptr
                                                    : &context.prefetched);
  if (!status) {
    result.status = status;
    return result;
  }
  if (context.host_token != 0u) {
    if (!context.pool->authority().complete(context.host_token, true)) {
      const bool host_clean =
          context.pool->prefetch[context.consume_lane].cancel(
              context.pool->authority(), context.prefetched, true, true);
      if (host_clean) {
        context.host_token = 0u;
      }
      result.status = Status::fail(Reason::PipelineInvalid);
      result.poison_pipeline = !host_clean;
      result.invalidate_all = true;
      return result;
    }
    const bool aliases_clean =
        context.pool->prefetch[context.consume_lane].release_aliases(
            context.pool->authority(), true);
    if (!aliases_clean) {
      result.status = Status::fail(Reason::PipelineInvalid);
      result.poison_pipeline = true;
      result.invalidate_all = true;
      return result;
    }
    context.host_token = 0u;
  }
  Accumulate(context.stats.pipeline.residency.stall_ns,
             pipeline_clock() - supply_started);
  Accumulate(context.stats.pipeline.residency.page_in_count, fetched_pages);

  if (!context.pool->authority().activate(context.execution_token)) {
    result.status = Status::fail(Reason::PipelineInvalid);
    result.poison_pipeline = true;
    result.invalidate_all = true;
    return result;
  }

  const std::uint64_t lookahead = context.run.active.stream.prefetch_distance();
  for (std::uint64_t distance = 1u; distance <= lookahead; ++distance) {
    if (context.epoch > std::numeric_limits<std::uint64_t>::max() - distance ||
        context.epoch + distance >= context.run.active.stream.epoch_count()) {
      break;
    }
    const std::size_t lane =
        static_cast<std::size_t>((context.epoch + distance) % 2u);
    if (context.prefetch_pending[lane]) {
      continue;
    }
    VirtualEpochProjection next{};
    if (!project_virtual_epoch(context.run, context.epoch + distance, next)) {
      result.status = Status::fail(Reason::PipelineInvalid);
      return result;
    }
    status = schedule_virtual_prefetch(
        context.input, next, context.run, *context.pool,
        context.pool->prefetch[lane], true, context.prefetch_pending[lane],
        context.cleanup_failed);
    if (!status) {
      result.status = status;
      result.poison_pipeline = context.cleanup_failed;
      return result;
    }
  }
  return result;
}

} // namespace rund::compute::detail
