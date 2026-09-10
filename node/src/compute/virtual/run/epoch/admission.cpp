#include "internal.hpp"

#include "../backing.hpp"

#include <limits>

namespace rund::compute::detail {

VirtualEpochResult admit_virtual_epoch(VirtualEpochContext &context) noexcept {
  residency::Pool *const fallback_pool =
      context.state.pipeline == nullptr
          ? nullptr
          : context.state.pipeline->residency_pool.get();
  if (!project_virtual_epoch(context.run, context.epoch, context.projected)) {
    const bool clean =
        cancel_virtual_prefetch(fallback_pool, context.prefetch_pending);
    return VirtualEpochResult{.status = Status::fail(Reason::PipelineInvalid),
                              .failed_page = context.projected.failed_page,
                              .poison_pipeline = !clean};
  }

  context.bank =
      static_cast<std::uint32_t>(context.epoch % residency::Pool::BankCount);
  context.selected = context.bank == 0u ? &context.state.pipeline
                                        : &context.state.alternate_pipeline;
  if (*context.selected == nullptr) {
    const bool clean =
        cancel_virtual_prefetch(fallback_pool, context.prefetch_pending);
    return VirtualEpochResult{.status = Status::fail(Reason::PipelineInvalid),
                              .failed_page = context.projected.failed_page,
                              .poison_pipeline = !clean};
  }
  context.pipeline = context.selected->get();
  if (context.pipeline->residency_pool == nullptr ||
      context.projected.page_count > PipelineLeafCapacity) {
    const bool clean =
        cancel_virtual_prefetch(fallback_pool, context.prefetch_pending);
    return VirtualEpochResult{.status = Status::fail(Reason::PipelineInvalid),
                              .failed_page = context.projected.failed_page,
                              .poison_pipeline = !clean};
  }
  context.pool = context.pipeline->residency_pool.get();
  context.consume_lane = static_cast<std::size_t>(context.epoch % 2u);
  if (context.pool->device == nullptr) {
    const bool clean =
        cancel_virtual_prefetch(context.pool, context.prefetch_pending);
    return VirtualEpochResult{.status = Status::fail(Reason::DeviceInvalid),
                              .failed_page = context.projected.failed_page,
                              .poison_pipeline = !clean};
  }
  if (context.pool->device->backend != Backend::Cpu &&
      !context.prefetch_pending[context.consume_lane]) {
    const Status scheduled = schedule_virtual_prefetch(
        context.input, context.projected, context.run, *context.pool,
        context.pool->prefetch[context.consume_lane], false,
        context.prefetch_pending[context.consume_lane], context.cleanup_failed);
    if (!scheduled) {
      const bool clean =
          cancel_virtual_prefetch(context.pool, context.prefetch_pending);
      return VirtualEpochResult{.status = scheduled,
                                .failed_page = context.projected.failed_page,
                                .poison_pipeline =
                                    context.cleanup_failed || !clean};
    }
  }
  context.prefetched = context.prefetch_pending[context.consume_lane]
                           ? context.pool->prefetch[context.consume_lane].wait()
                           : residency::PrefetchReceipt{};
  context.prefetch_pending[context.consume_lane] = false;
  context.host_token = context.prefetched.token;
  if (!context.prefetched.status) {
    bool clean = true;
    if (context.host_token != 0u) {
      clean = context.pool->prefetch[context.consume_lane].cancel(
          context.pool->authority(), context.prefetched, true, true);
    }
    clean = cancel_virtual_prefetch(context.pool, context.prefetch_pending) &&
            clean;
    return VirtualEpochResult{.status = context.prefetched.status,
                              .failed_page = context.projected.failed_page,
                              .poison_pipeline = !clean};
  }

  context.epoch_pages = static_cast<std::size_t>(context.projected.page_count);
  if (!project_residency_transform_uses(
          context.projected, context.run,
          std::span<residency::CacheUse>{context.input_uses.data(),
                                         context.epoch_pages},
          std::span<residency::CacheUse>{context.output_uses.data(),
                                         context.epoch_pages})) {
    bool clean =
        cancel_virtual_prefetch(context.pool, context.prefetch_pending);
    if (context.host_token != 0u) {
      clean = context.pool->prefetch[context.consume_lane].cancel(
                  context.pool->authority(), context.prefetched, true, true) &&
              clean;
    }
    return VirtualEpochResult{.status =
                                  Status::fail(clean ? Reason::PipelineInvalid
                                                     : Reason::PipelineBusy),
                              .failed_page = context.projected.failed_page,
                              .poison_pipeline = true};
  }
  const residency::FrameTier execution_tier =
      context.pool->device->backend == Backend::Cpu
          ? residency::FrameTier::Host
          : residency::FrameTier::Device;
  context.acquired = context.pool->authority().begin_transform(
      std::span<const residency::CacheUse>{context.input_uses.data(),
                                           context.epoch_pages},
      context.pool->input_regions[context.bank],
      std::span<const residency::CacheUse>{context.output_uses.data(),
                                           context.epoch_pages},
      residency::FrameRegion{
          .tier = execution_tier,
          .role = residency::FrameRole::Output,
          .first = context.pool->first_output_frame +
                   context.bank *
                       static_cast<std::uint32_t>(context.run.frame_capacity),
          .count = static_cast<std::uint32_t>(context.run.frame_capacity),
      });
  if (!context.acquired) {
    bool clean =
        cancel_virtual_prefetch(context.pool, context.prefetch_pending);
    if (context.host_token != 0u) {
      const bool host_published =
          context.pool->authority().complete(context.host_token, true);
      bool host_clean = host_published;
      if (!host_published) {
        host_clean = context.pool->prefetch[context.consume_lane].cancel(
            context.pool->authority(), context.prefetched, true, true);
      } else {
        host_clean =
            context.pool->prefetch[context.consume_lane].release_aliases(
                context.pool->authority(), true);
      }
      clean = host_published && host_clean && clean;
    }
    if (!clean) {
      return VirtualEpochResult{.status = Status::fail(Reason::PipelineInvalid),
                                .failed_page = context.projected.failed_page,
                                .poison_pipeline = true};
    }
    return VirtualEpochResult{
        .status = Status::fail(context.acquired.failure ==
                                       residency::AuthorityFailure::Busy
                                   ? Reason::PipelineBusy
                                   : Reason::PipelineMemoryBudget),
        .failed_page = context.projected.failed_page};
  }

  context.execution_token = context.acquired.lease.token;
  if (context.acquired.lease.bindings.size() != context.epoch_pages * 2u) {
    const bool clean = context.pool->authority().complete(
        context.execution_token, false, true);
    context.execution_token = 0u;
    return VirtualEpochResult{.status = Status::fail(Reason::PipelineInvalid),
                              .failed_page = context.projected.failed_page,
                              .poison_pipeline = !clean};
  }
  context.input_lease = residency::EpochLease{
      .bindings = context.acquired.lease.bindings.first(context.epoch_pages),
      .transitions = context.acquired.lease.transitions,
      .token = context.acquired.lease.token,
  };
  context.device_output_lease = residency::EpochLease{
      .bindings = context.acquired.lease.bindings.subspan(context.epoch_pages,
                                                          context.epoch_pages),
      .token = context.acquired.lease.token,
  };
  return {};
}

} // namespace rund::compute::detail
