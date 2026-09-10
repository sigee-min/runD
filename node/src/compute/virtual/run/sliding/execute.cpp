#include "internal.hpp"

namespace rund::compute::detail {

VirtualExecutionResult execute_accel_virtual_execution_sliding(
    VirtualPipelineState &state, VirtualBacking &input, VirtualBacking &output,
    const VirtualRunProjection &run,
    const VirtualExecutionSlidingPrepared &prepared, Stats &stats) noexcept {
  using namespace sliding_product_detail;
  VirtualExecutionResult result{};
  const auto owner = validate_execution_owner(state, prepared);
  if (owner == nullptr) {
    // Prepared-owner authentication failure is terminal. It is not evidence
    // that the backend declined before touching the native route.
    result.status = Status::fail(Reason::BackendUnsupported);
    result.disposition = VirtualExecutionDisposition::Terminal;
    return result;
  }
  SlidingProductRun &wait = *owner->run;
  bool idle = false;
  {
    std::lock_guard run_lock{wait.gate};
    if (wait.quarantine == nullptr &&
        !wait.poison.load(std::memory_order_acquire)) {
      std::lock_guard control_lock{wait.persistent_control.gate};
      idle = !wait.persistent_control.active &&
             wait.persistent_control.native == nullptr &&
             wait.sliding.quiescent();
    }
  }
  if (!idle) {
    result.status = Status::fail(Reason::DeviceLost);
    result.poison_pipeline = true;
    return result;
  }
  {
    std::lock_guard lock{wait.gate};
    wait.persistent_final = {};
    wait.persistent_final_received = false;
  }
  const Status memory = renew_persistent_memory(state, *owner, wait);
  if (!memory) {
    result.status = memory;
    return result;
  }
  residency::Pool &pool = *state.pipeline->residency_pool;
  auto sliding = pool.authority().sliding();
  residency::ExecutionLease lease =
      sliding.begin_execution_sliding(owner->plan);
  if (!lease) {
    result.status =
        Status::fail(lease.failure == residency::AuthorityFailure::Busy
                         ? Reason::PipelineBusy
                         : Reason::PipelineMemoryBudget);
    return result;
  }
  residency::execution::Sliding joined = bind_controller(pool, *owner, lease);
  if (!joined) {
    result.status = Status::fail(Reason::PipelineMemoryBudget);
    return result;
  }
  if (!sliding.seal_execution_sliding_lease(lease)) {
    const bool closed =
        sliding.abandon_bound_execution_sliding(owner->plan, joined, lease);
    result.status = Status::fail(Reason::PipelineInvalid);
    result.poison_pipeline = !closed;
    return result;
  }
  const bool cold_lowering =
      wait.persistent_preparation.backend.lowering == nullptr;
  const Status lowered = prepare_persistent_lowering(wait, *owner, pool, lease);
  if (!lowered) {
    const bool closed =
        sliding.abandon_bound_execution_sliding(owner->plan, joined, lease);
    result.status = lowered;
    result.poison_pipeline = !closed;
    // This is the sole execution-side capability fallback proof: lowering
    // declined before pipeline initialization/native acceptance and the
    // temporary Authority lease closed cleanly. Every later unsupported
    // status remains terminal, even when poison_pipeline is false.
    if (cold_lowering && closed &&
        lowered.reason() == Reason::BackendUnsupported) {
      result.disposition = VirtualExecutionDisposition::CleanPreNativeDecline;
    }
    return result;
  }

  initialize_run(wait, *owner, prepared.owner, state, input, output, run, stats,
                 pool, std::move(joined));
  const Status pipeline = begin_pipeline_sliding(wait);
  if (!pipeline) {
    const bool closed = close_rejected_start(wait, pool, *owner, pipeline);
    if (closed) {
      result.status = pipeline;
      release_run_links(wait);
      return result;
    }
    quarantine_final(wait);
    result.status = Status::fail(Reason::DeviceLost);
    result.poison_pipeline = true;
    return result;
  }

  const std::uint64_t submitted_ns = pipeline_clock();
  const Status submitted = submit_persistent(wait);
  if (!submitted) {
    if (persistent_submission_started(wait)) {
      complete_persistent_product(wait, Status::fail(Reason::DeviceLost));
      {
        std::lock_guard lock{wait.gate};
        result = wait.result;
      }
      result.poison_pipeline = true;
      return result;
    }
    const bool closed = close_rejected_start(wait, pool, *owner, submitted);
    static_cast<void>(finish_pipeline_sliding(wait, submitted));
    result.status =
        closed ? submitted : Status::fail(Reason::CompletionInvalid);
    result.poison_pipeline = !closed;
    if (closed) {
      release_run_links(wait);
    }
    return result;
  }
  const Status serviced = service_persistent_recurrence(wait);
  complete_persistent_product(wait, serviced);
  {
    std::lock_guard lock{wait.gate};
    result = wait.result;
  }
  stats.pipeline.residency.stall_ns = std::min(
      std::numeric_limits<std::uint64_t>::max(),
      stats.pipeline.residency.stall_ns + (pipeline_clock() - submitted_ns));
  if (!result.poison_pipeline) {
    release_run_links(wait);
  }
  return result;
}

} // namespace rund::compute::detail
