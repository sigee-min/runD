#include "local.hpp"
#include "../../../../backend.hpp"

#include "../../../../pipeline/local.hpp"
#include "../../../../pipeline/run/clock.hpp"
#include "../../backing.hpp"
#include "../../cache.hpp"

#include <algorithm>
#include <span>

namespace rund::compute::detail::virtual_run_overlap::prepare_detail {

Status admit(Context &context) noexcept {
  context.prepared = {};
  context.prepared.ordinal = context.epoch;
  if (!project_virtual_epoch(context.run, context.epoch,
                             context.prepared.projection) ||
      context.prepared.projection.page_count > PipelineLeafCapacity) {
    return Status::fail(Reason::PipelineInvalid);
  }

  const std::uint32_t bank =
      static_cast<std::uint32_t>(context.epoch % residency::Pool::BankCount);
  context.prepared.pipeline =
      bank == 0u ? context.state.pipeline : context.state.alternate_pipeline;
  if (context.prepared.pipeline == nullptr ||
      context.prepared.pipeline->residency_pool == nullptr ||
      context.prepared.pipeline->residency_bank != bank) {
    return Status::fail(Reason::PipelineInvalid);
  }
  context.pool = context.prepared.pipeline->residency_pool.get();
  context.lane = static_cast<std::size_t>(context.epoch % 2u);
  context.ready.started = pipeline_clock();
  if (context.pool->device == nullptr) {
    return Status::fail(Reason::DeviceInvalid);
  }

  context.coherent_input_capable =
      is_accelerator(context) && !context.run.scan && !context.run.reduction &&
      !context.run.graph_reduction &&
      context.run.host_frame_capacity == context.run.frame_capacity &&
      context.prepared.pipeline->residency_stage ==
          PipelineResidencyStage::Direct &&
      context.pool->device->ops != nullptr &&
      context.pool->device->ops->host_write != nullptr;
  if (is_accelerator(context) && !context.prefetch_pending[context.lane]) {
    const BufferWriteView view =
        context.coherent_input_capable
            ? residency_input_view(*context.prepared.pipeline, context.run)
            : BufferWriteView{};
    const Status scheduled = schedule_virtual_prefetch(
        context.input, context.prepared.projection, context.run, *context.pool,
        context.pool->prefetch[context.lane], false,
        context.prefetch_pending[context.lane], context.poison,
        view ? std::span<std::byte>{view.data, view.bytes}
             : std::span<std::byte>{});
    if (!scheduled) {
      return scheduled;
    }
  }

  context.prefetched = context.prefetch_pending[context.lane]
                           ? context.pool->prefetch[context.lane].wait()
                           : residency::PrefetchReceipt{};
  context.prefetch_pending[context.lane] = false;
  context.host_token = context.prefetched.token;
  if (!context.prefetched.status) {
    const bool cancelled =
        cancel_prefetch_receipt(context, context.prefetched, true);
    context.poison = !cancelled || context.poison;
    return context.prefetched.status;
  }

  const std::size_t epoch_pages =
      static_cast<std::size_t>(context.prepared.projection.page_count);
  residency::AuthorityResult acquired{};
  if (context.prefetched.coherent_input) {
    acquired = context.pool->authority().resume(context.host_token);
  } else {
    std::array<residency::CacheUse, PipelineLeafCapacity> input_uses{};
    std::array<residency::CacheUse, PipelineLeafCapacity> output_uses{};
    if (!project_residency_transform_uses(
            context.prepared.projection, context.run,
            std::span<residency::CacheUse>{input_uses.data(), epoch_pages},
            std::span<residency::CacheUse>{output_uses.data(), epoch_pages})) {
      const bool cancelled =
          cancel_prefetch_receipt(context, context.prefetched, true);
      context.poison = !cancelled || context.poison;
      context.poison = true;
      return Status::fail(Reason::PipelineInvalid);
    }
    const residency::FrameTier execution_tier =
        is_accelerator(context) ? residency::FrameTier::Device
                                : residency::FrameTier::Host;
    acquired = context.pool->authority().begin_transform(
        std::span<const residency::CacheUse>{input_uses.data(), epoch_pages},
        context.pool->input_regions[bank],
        std::span<const residency::CacheUse>{output_uses.data(), epoch_pages},
        residency::FrameRegion{
            .tier = execution_tier,
            .role = residency::FrameRole::Output,
            .first =
                context.pool->first_output_frame +
                bank * static_cast<std::uint32_t>(context.run.frame_capacity),
            .count = static_cast<std::uint32_t>(context.run.frame_capacity),
        });
  }

  if (!acquired || acquired.lease.bindings.size() != epoch_pages * 2u ||
      acquired.lease.bindings.size() > context.prepared.bindings.size() ||
      acquired.lease.transitions.size() > context.prepared.transitions.size()) {
    bool clean = true;
    if (context.prefetched.coherent_input && context.host_token != 0u) {
      clean = cancel_prefetch_receipt(context, context.prefetched, true);
      if (clean) {
        context.host_token = 0u;
      }
    } else if (acquired) {
      clean =
          context.pool->authority().complete(acquired.lease.token, false, true);
    }
    if (!context.prefetched.coherent_input && context.host_token != 0u) {
      const bool host_published =
          context.pool->authority().complete(context.host_token, true);
      bool host_clean = host_published;
      if (!host_published) {
        host_clean = cancel_prefetch_receipt(context, context.prefetched, true);
      } else {
        host_clean = release_prefetch_aliases(context, true);
      }
      clean = host_published && host_clean && clean;
      if (host_clean) {
        context.host_token = 0u;
      }
    }
    if (!clean) {
      clean = release_prefetch_aliases(context, true) && clean;
      context.poison = true;
      return Status::fail(Reason::PipelineInvalid);
    }
    clean = release_prefetch_aliases(context, true) && clean;
    return Status::fail(acquired.failure == residency::AuthorityFailure::Busy
                            ? Reason::PipelineBusy
                            : Reason::PipelineMemoryBudget);
  }

  context.prepared.token = acquired.lease.token;
  context.prepared.coherent_input = context.prefetched.coherent_input;
  if (context.prepared.coherent_input) {
    context.host_token = 0u;
  }
  context.prepared.binding_count = epoch_pages;
  context.prepared.transition_count = acquired.lease.transitions.size();
  std::copy(acquired.lease.bindings.begin(), acquired.lease.bindings.end(),
            context.prepared.bindings.begin());
  std::copy(acquired.lease.transitions.begin(),
            acquired.lease.transitions.end(),
            context.prepared.transitions.begin());
  for (std::size_t index = 0u; index < context.prepared.binding_count;
       ++index) {
    context.prepared.keys[index] =
        context.prepared.bindings[context.prepared.binding_count + index].key;
  }
  if (std::any_of(
          acquired.lease.transitions.begin(), acquired.lease.transitions.end(),
          [](const residency::CacheTransition transition) {
            return transition.kind == residency::TransitionKind::Writeback;
          })) {
    return abort(context, Status::fail(Reason::PipelineInvalid));
  }
  return Status::success();
}

} // namespace rund::compute::detail::virtual_run_overlap::prepare_detail
