#include "internal.hpp"

#include "../../backing.hpp"
#include "projection.hpp"

namespace rund::compute::detail::graph_reduce {

Status initialize_ticket(GraphExecutionContext &context, Ticket &ticket,
                         const std::uint64_t batch) noexcept {
  if (!context.wavefront.admit(batch)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const Status projected =
      project_ticket(context.state, context.run, context.graph, context.pool,
                     batch, context.terminal_stage, context.capacity, ticket);
  if (projected && context.state.cpu_receipts != nullptr) {
    ticket.book_domain = context.state.cpu_receipts->domain();
  }
  return projected;
}

VirtualGraphResult prepare_initial(GraphExecutionContext &context) noexcept {
  // This check intentionally precedes any ticket admission, matching the
  // original boundary: there is no active owner to close on these failures.
  if (context.batches == 0u || context.capacity == 0u ||
      context.identity == 0u) {
    const Status failure = Status::fail(Reason::PipelineInvalid);
    context.state.failure_log.note(Fail::NoStage, 0u, Phase::Prepare,
                                   Check::Prepare, failure);
    return failed(failure, 0u, true);
  }
  if (context.persisted_output) {
    VirtualBackingAccess::require_recovery(*context.output,
                                           context.run.active.output_bytes);
  }

  Ticket &initial = context.tickets[0];
  Status status = initialize_ticket(context, initial, 0u);
  if (status && !context.cpu) {
    const std::uint64_t distance = context.run.active.graph.prefetch_distance();
    if (distance > PrefetchController::LaneCount) {
      status = Status::fail(Reason::PipelineInvalid);
    } else {
      status =
          context.prefetch.schedule(0u, false, context.prefetch_cleanup_failed);
      if (status && distance != 0u && context.batches > 1u &&
          !context.parallel_route) {
        status = context.prefetch.schedule(1u, true,
                                           context.prefetch_cleanup_failed);
      }
    }
  }
  if (!status) {
    return finish_failure(context, status, initial.pages.first_page,
                          context.prefetch_cleanup_failed, Phase::Prepare,
                          Check::Prepare, 0u, initial.batch);
  }
  status = context.supply.prepare(initial, nullptr, false);
  if (status) {
    status = context.supply.make_device_ready(initial, nullptr);
  }
  if (status) {
    status = context.prefetch.replenish(initial.batch,
                                        context.prefetch_cleanup_failed);
  }
  if (status) {
    status = context.stages.submit(initial, ExecutionStage::Prefix);
  }
  if (!status) {
    return finish_failure(context, status, initial.pages.first_page,
                          context.prefetch_cleanup_failed, Phase::Supply,
                          Check::Supply, 0u, initial.batch);
  }
  return {};
}

} // namespace rund::compute::detail::graph_reduce
