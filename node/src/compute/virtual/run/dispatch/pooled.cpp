#include "local.hpp"


namespace rund::compute::detail::virtual_run_dispatch {

VirtualRunDispatchResult dispatch_pooled(
    VirtualPipelineState &state, const std::span<VirtualBacking *const> inputs,
    VirtualBacking &input, VirtualBacking &output, VirtualRunProjection &run,
    const VirtualDeviceVsmCandidate &candidate,
    const std::uint64_t active_count, Stats &stats, VirtualRunWork &work,
    VirtualRunTransaction &transaction, VirtualRunResources &resources,
    bool &poison_pipeline) noexcept {
  const Status pool_locked = lock_pool(state, resources);
  if (!pool_locked) {
    return VirtualRunDispatchResult{.status = pool_locked,
                                    .poison_pipeline = poison_pipeline,
                                    .direct_terminal = true};
  }
  const Status prepared = prepare_work(run, work);
  if (!prepared) {
    return VirtualRunDispatchResult{.status = prepared,
                                    .poison_pipeline = poison_pipeline,
                                    .direct_terminal = true};
  }
  if (active_count == 0u) {
    const Status staged =
        stage_pool(state, run, stats, resources, poison_pipeline);
    if (!staged) {
      return VirtualRunDispatchResult{.status = staged,
                                      .poison_pipeline = poison_pipeline,
                                      .direct_terminal = true};
    }
    const Status closed = close_pool_stage(resources);
    if (!closed) {
      poison_pipeline = true;
      return VirtualRunDispatchResult{
          .status = closed,
          .poison_pipeline = true,
          .direct_terminal = true,
          .certainty = VirtualRunWriteCertainty::UnknownMayWrite,
      };
    }
    return VirtualRunDispatchResult{.status = Status::success(),
                                    .direct_terminal = true,
                                    .empty = true,
                                    .reduction_pending = run.reduction()};
  }

  VirtualDeviceVsmPostStage post{};
  const bool selected = candidate.device_vsm();
  if (selected) {
    post = prepare_virtual_device_vsm_route(state, run, candidate);
    if (!post.ready()) {
      return dispose_device_vsm_route(state, resources, post,
                                      VirtualDeviceVsmScope::Pooled,
                                      post.status, poison_pipeline);
    }
  }
  const Status staged =
      stage_pool(state, run, stats, resources, poison_pipeline);
  if (!staged) {
    if (selected) {
      return dispose_device_vsm_route(state, resources, post,
                                      VirtualDeviceVsmScope::Pooled, staged,
                                      poison_pipeline);
    }
    return VirtualRunDispatchResult{.status = staged,
                                    .poison_pipeline = poison_pipeline,
                                    .direct_terminal = true};
  }
  const std::uint64_t page_count = resources.page_count;
  if (selected) {
    return execute_device_vsm_stage(state, inputs, output, run, post,
                                    VirtualDeviceVsmScope::Pooled, stats,
                                    resources, poison_pipeline);
  }

  const Status closed = close_pool_stage(resources);
  if (!closed) {
    poison_pipeline = true;
    return VirtualRunDispatchResult{
        .status = closed,
        .poison_pipeline = true,
        .direct_terminal = true,
        .certainty = VirtualRunWriteCertainty::UnknownMayWrite,
    };
  }
  const VirtualRunAdmission admission =
      admit_virtual_run(state, run, input, output);
  const Status transaction_ready = begin_virtual_run_transaction(
      state, admission, run, output, page_count, transaction);
  if (!transaction_ready) {
    return VirtualRunDispatchResult{.status = transaction_ready,
                                    .poison_pipeline = poison_pipeline,
                                    .direct_terminal = true};
  }
  if (run.scan() && !run.graph_execution() && !run.device_vsm_required) {
    const Status cursor_ready =
        begin_virtual_run_publication_cursor(state, transaction);
    if (!cursor_ready) {
      return VirtualRunDispatchResult{.status = cursor_ready,
                                      .poison_pipeline = poison_pipeline,
                                      .direct_terminal = true};
    }
  }
  if (!run.graph_execution() && !bind_virtual_run_transfer(state, run)) {
    poison_pipeline = true;
    return VirtualRunDispatchResult{
        .status = Status::fail(Reason::PipelineInvalid),
        .poison_pipeline = true,
        .direct_terminal = true,
    };
  }
  return dispatch_run(state, inputs, input, output, run, admission, stats, work,
                      &transaction);
}

} // namespace rund::compute::detail::virtual_run_dispatch
