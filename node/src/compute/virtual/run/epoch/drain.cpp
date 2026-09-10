#include "internal.hpp"

#include <span>

namespace rund::compute::detail {

VirtualEpochPhaseResult
drain_virtual_epoch_output(VirtualEpochContext &context) noexcept {
  VirtualEpochPhaseResult result{};
  const auto begin_device_drain = [&]() noexcept {
    const std::span<const residency::CacheKey> keys{context.dirty.data(),
                                                    context.dirty_count};
    if (context.pipeline->device->backend != Backend::Cpu) {
      return context.pool->authority().begin_migration(
          keys, context.pool->first_output_frame,
          context.pool->output_frame_count);
    }
    return context.run.reduction ? context.pool->authority().begin_discard(
                                       keys, context.pool->first_output_frame,
                                       context.pool->output_frame_count)
                                 : context.pool->authority().begin_writeback(
                                       keys, context.pool->first_output_frame,
                                       context.pool->output_frame_count);
  };
  residency::AuthorityResult writeback = begin_device_drain();
  if (!writeback) {
    bool cleaned =
        cancel_residency_output(*context.pipeline, context.output_reservation);
    cleaned = cancel_virtual_prefetch(context.pool, context.prefetch_pending) &&
              cleaned;
    const residency::AuthorityResult retry = begin_device_drain();
    cleaned = retry && context.pool->authority().discard(retry.lease.token) &&
              cleaned;
    result.status =
        Status::fail(cleaned ? Reason::PipelineInvalid : Reason::PipelineBusy);
    result.poison_pipeline = true;
    return result;
  }
  if (context.pipeline->device->backend != Backend::Cpu) {
    const std::uint64_t device_token = writeback.lease.token;
    const std::uint64_t output_token = context.output_reservation.token;
    const bool migrated = context.pool->authority().complete_migration(
        device_token, output_token);
    if (!migrated) {
      const bool device_terminal =
          context.pool->authority().discard(device_token);
      const bool output_terminal = cancel_residency_output(
          *context.pipeline, context.output_reservation);
      const bool prefetch_terminal =
          cancel_virtual_prefetch(context.pool, context.prefetch_pending);
      result.status =
          Status::fail(device_terminal && output_terminal && prefetch_terminal
                           ? Reason::PipelineInvalid
                           : Reason::PipelineBusy);
      result.poison_pipeline = true;
      return result;
    }
    context.output_reservation.token = 0u;
    std::array<residency::CacheKey, PipelineLeafCapacity> output_keys{};
    for (std::size_t index = 0u; index < context.output_lease.bindings.size();
         ++index) {
      output_keys[index] = context.output_lease.bindings[index].key;
    }
    const std::span<const residency::CacheKey> host_keys{
        output_keys.data(), context.output_lease.bindings.size()};
    writeback = context.run.reduction
                    ? context.pool->authority().begin_discard(
                          host_keys, context.pool->first_host_output_frame,
                          context.pool->host_output_frame_count)
                    : context.pool->authority().begin_writeback(
                          host_keys, context.pool->first_host_output_frame,
                          context.pool->host_output_frame_count);
    if (!writeback) {
      const bool prefetch_terminal =
          cancel_virtual_prefetch(context.pool, context.prefetch_pending);
      const residency::AuthorityResult retry =
          context.run.reduction
              ? context.pool->authority().begin_discard(
                    host_keys, context.pool->first_host_output_frame,
                    context.pool->host_output_frame_count)
              : context.pool->authority().begin_writeback(
                    host_keys, context.pool->first_host_output_frame,
                    context.pool->host_output_frame_count);
      const bool output_terminal =
          retry && context.pool->authority().discard(retry.lease.token);
      result.status = Status::fail(prefetch_terminal && output_terminal
                                       ? Reason::PipelineInvalid
                                       : Reason::PipelineBusy);
      result.poison_pipeline = true;
      return result;
    }
  }
  if (context.run.reduction) {
    const bool completed =
        context.pool->authority().discard(writeback.lease.token);
    if (!completed) {
      const bool prefetch_clean =
          cancel_virtual_prefetch(context.pool, context.prefetch_pending);
      result.status = Status::fail(prefetch_clean ? Reason::PipelineInvalid
                                                  : Reason::PipelineBusy);
      result.poison_pipeline = true;
      return result;
    }
    return result;
  }
  Status status = writeback_residency_cache(
      context.output, context.run, writeback.lease.transitions,
      context.stats.pipeline.residency, nullptr, {}, context.transaction);
  const bool published =
      status && context.pool->authority().complete(writeback.lease.token, true);
  bool completed =
      status ? published
             : context.pool->authority().discard(writeback.lease.token);
  if (status && !published) {
    completed = context.pool->authority().discard(writeback.lease.token);
  }
  if (!status || !published || !completed) {
    const bool prefetch_clean =
        cancel_virtual_prefetch(context.pool, context.prefetch_pending);
    result.status = status ? Status::fail(Reason::PipelineInvalid) : status;
    result.poison_pipeline =
        (status && !published) || !completed || !prefetch_clean;
    return result;
  }
  return result;
}

} // namespace rund::compute::detail
