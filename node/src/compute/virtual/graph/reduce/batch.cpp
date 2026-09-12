#include "internal.hpp"

#include "../../backing.hpp"

namespace rund::compute::detail::graph_reduce {

namespace {

[[nodiscard]] VirtualGraphResult
prepare_next(GraphExecutionContext &context, Ticket &current, Ticket &next,
             const std::uint64_t batch) noexcept {
  Status status = context.persists.reuse(next, context.child_poison);
  if (status) {
    status = initialize_ticket(context, next, batch + 1u);
  }
  if (status) {
    status = context.supply.prepare(next, &current.prefix_timeline, true);
  }
  if (!status) {
    return finish_failure(context, status, next.pages.first_page, false,
                          Phase::Prepare, Check::Prepare, Fail::NoStage,
                          batch + 1u);
  }
  return {};
}

[[nodiscard]] VirtualGraphResult
ready_next(GraphExecutionContext &context, Ticket &current, Ticket &next,
           const std::uint64_t batch) noexcept {
  Status status = context.persists.reuse(next, context.child_poison);
  if (status) {
    status = initialize_ticket(context, next, batch + 1u);
  }
  if (status) {
    status = context.supply.prepare(next, &current.prefix_timeline, true);
  }
  if (status) {
    status =
        context.supply.make_device_ready(next, &current.collective_timeline);
  }
  if (status) {
    status =
        context.prefetch.replenish(next.batch, context.prefetch_cleanup_failed);
  }
  if (status) {
    status = context.stages.submit(next, ExecutionStage::Prefix);
  }
  if (!status) {
    return finish_failure(
        context, status, next.pages.first_page,
        context.prefetch_cleanup_failed || context.child_poison, Phase::Supply,
        Check::Supply, Fail::NoStage, batch + 1u);
  }
  return {};
}

} // namespace

VirtualGraphResult execute_batch(GraphExecutionContext &context,
                                 const std::uint64_t batch) noexcept {
  Ticket &current = context.tickets[static_cast<std::size_t>(
      batch % residency::Pool::BankCount)];
  const bool has_next = batch + 1u < context.batches;
  Ticket *next = has_next ? &context.tickets[static_cast<std::size_t>(
                                (batch + 1u) % residency::Pool::BankCount)]
                          : nullptr;
  if (current.batch != batch || current.phase != TicketPhase::PrefixRunning) {
    const Status failure = Status::fail(Reason::PipelineInvalid);
    return finish_failure(context, failure, current.pages.first_page, true,
                          Phase::Prepare, Check::Ticket, Fail::NoStage, batch);
  }

  const bool defer_next =
      next != nullptr && next->phase == TicketPhase::OutputPersisting;
  bool next_prepared = false;
  if (next != nullptr && !defer_next && !context.parallel_route) {
    const VirtualGraphResult prepared =
        prepare_next(context, current, *next, batch);
    if (!prepared.status) {
      return prepared;
    }
    next_prepared = true;
  }

  Status status = finish_prefix(
      context.authority, context.graph, context.wavefront, context.stages,
      current, context.state.failure_log, context.child_poison);
  if (!status) {
    return finish_failure(context, status, current.pages.first_page,
                          context.child_poison, Phase::Prefix, Check::Execute,
                          0u, batch);
  }
  status = context.middle.execute(current, context.child_poison);
  if (!status) {
    return finish_failure(context, status, current.pages.first_page,
                          context.child_poison, Phase::Middle, Check::Execute,
                          Fail::NoStage, batch);
  }
  status = context.collective.prepare(current);
  if (status) {
    status = context.stages.submit(current, ExecutionStage::Collective);
  }
  if (!status) {
    return finish_failure(
        context, status, current.pages.first_page, context.child_poison,
        Phase::Collective, Check::Submit,
        static_cast<std::uint32_t>(context.terminal_stage), batch);
  }

  if (next_prepared) {
    status =
        context.supply.make_device_ready(*next, &current.collective_timeline);
    if (status) {
      status = context.prefetch.replenish(next->batch,
                                          context.prefetch_cleanup_failed);
    }
    if (!status) {
      return finish_failure(
          context, status, next->pages.first_page,
          context.prefetch_cleanup_failed || context.child_poison,
          Phase::Supply, Check::Supply, Fail::NoStage, batch + 1u);
    }
  }

  status = context.collective.finish(current, context.child_poison);
  if (status) {
    status = current.collective->device->backend == Backend::Cpu
                 ? Status::success()
                 : issue_output_drain(context.state, context.pool, context.run,
                                      context.graph, context.terminal_stage,
                                      current);
  }
  if (!status) {
    return finish_failure(
        context, status, current.pages.first_page, context.child_poison,
        Phase::Collective, Check::Execute,
        static_cast<std::uint32_t>(context.terminal_stage), batch);
  }
  if (next_prepared) {
    status = context.stages.submit(*next, ExecutionStage::Prefix);
    if (!status) {
      return finish_failure(context, status, next->pages.first_page,
                            context.child_poison, Phase::Prefix, Check::Submit,
                            0u, batch + 1u);
    }
  }
  if (context.persisted_output) {
    status = context.persists.before_start(context.tickets, current,
                                           context.child_poison);
    if (status) {
      status = context.persists.start(
          current, next_prepared ? &next->prefix_timeline : nullptr,
          context.child_poison);
    }
  } else {
    status = finish_output(
        context.authority, context.run, *context.reduction, context.stats,
        context.wavefront, context.stages, current,
        next_prepared ? &next->prefix_timeline : nullptr, context.child_poison);
  }
  if (!status) {
    return finish_failure(
        context, status, current.pages.first_page, context.child_poison,
        context.persisted_output ? Phase::Persist : Phase::Output,
        context.persisted_output ? Check::Persist : Check::Close,
        static_cast<std::uint32_t>(context.terminal_stage), batch);
  }
  if (defer_next) {
    const VirtualGraphResult ready = ready_next(context, current, *next, batch);
    if (!ready.status) {
      return ready;
    }
  }
  if (context.parallel_route && next != nullptr && !defer_next) {
    const VirtualGraphResult ready = ready_next(context, current, *next, batch);
    if (!ready.status) {
      return ready;
    }
  }
  return {};
}

VirtualGraphResult execute_batches(GraphExecutionContext &context) noexcept {
  for (std::uint64_t batch = 0u; batch < context.batches; ++batch) {
    const VirtualGraphResult result = execute_batch(context, batch);
    if (!result.status) {
      return result;
    }
  }
  return {};
}

} // namespace rund::compute::detail::graph_reduce
