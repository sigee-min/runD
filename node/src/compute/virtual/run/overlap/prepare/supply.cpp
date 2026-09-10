#include "local.hpp"

#include "../../../../pipeline/run/clock.hpp"
#include "../../backing.hpp"
#include "../../cache.hpp"

#include <rund/counter.hpp>

namespace rund::compute::detail::virtual_run_overlap::prepare_detail {

Status supply(Context &context) noexcept {
  const VirtualSupplyResult supplied = read_virtual_epoch(
      context.input, context.prepared.projection, context.run,
      input_lease(context.prepared), context.prefetched,
      context.stats.pipeline.residency, context.input_reuse);
  if (!supplied.status) {
    return abort(context, supplied.status);
  }

  std::uint64_t hits = 0u;
  for (std::size_t index = 0u; index < context.prepared.binding_count;
       ++index) {
    context.fetched +=
        static_cast<std::uint64_t>(context.prepared.bindings[index].fetch);
    hits += static_cast<std::uint64_t>(!context.prepared.bindings[index].fetch);
  }
  std::uint64_t evictions = 0u;
  const std::uint32_t bank = context.prepared.pipeline->residency_bank;
  const residency::FrameRegion input_region = context.pool->input_regions[bank];
  const std::uint64_t input_bank_first = input_region.first;
  for (std::size_t index = 0u; index < context.prepared.transition_count;
       ++index) {
    const residency::CacheTransition &transition =
        context.prepared.transitions[index];
    evictions += static_cast<std::uint64_t>(
        transition.kind == residency::TransitionKind::Unmap &&
        transition.frame >= input_bank_first &&
        transition.frame - input_bank_first < context.run.frame_capacity);
  }
  if (supplied.fetched_pages != context.fetched) {
    return abort(context, Status::fail(Reason::PipelineInvalid));
  }

  using ::rund::detail::counter::Accumulate;
  Accumulate(context.stats.pipeline.residency.cache_hit_count, hits);
  Accumulate(context.stats.pipeline.residency.eviction_count, evictions);
  Accumulate(context.stats.pipeline.residency.late_page_count,
             supplied.late_pages);
  Accumulate(context.stats.pipeline.residency.page_in_bytes,
             is_accelerator(context)
                 ? context.fetched * context.run.input_page_bytes
                 : supplied.backing_bytes);

  VirtualTransferInterval upload_interval{};
  const residency::PrefetchReceipt *prefetched_transfer =
      is_accelerator(context) ? &context.prefetched : nullptr;
  const Status uploaded = supply_residency_cache(
      *context.prepared.pipeline, context.run, input_lease(context.prepared),
      context.stats, prefetched_transfer, &upload_interval);
  context.ready.completed = pipeline_clock();
  if (!uploaded) {
    return abort(context, uploaded);
  }

  if (context.host_token != 0u) {
    if (!context.pool->authority().complete(context.host_token, true)) {
      const bool cancelled =
          cancel_prefetch_receipt(context, context.prefetched, true);
      context.poison = !cancelled || context.poison;
      if (cancelled) {
        context.host_token = 0u;
      }
      return abort(context, Status::fail(Reason::PipelineInvalid));
    }
    context.poison = !release_prefetch_aliases(context, true) || context.poison;
    context.host_token = 0u;
  }
  if (context.fetched != 0u) {
    context.upload = TimelineInterval{
        .started = upload_interval.started_ns,
        .completed = upload_interval.completed_ns,
    };
  }
  Accumulate(context.stats.pipeline.residency.page_in_count, context.fetched);

  if (!context.prepared.coherent_input &&
      !context.pool->authority().activate(context.prepared.token)) {
    return abort(context, Status::fail(Reason::PipelineInvalid));
  }
  return Status::success();
}

} // namespace rund::compute::detail::virtual_run_overlap::prepare_detail
