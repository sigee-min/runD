#include "internal.hpp"

#include <thread>

namespace rund::compute::detail::graph_reduce {

MiddleController::SelectResult MiddleController::select(
    Ticket &ticket, StageScratch &scratch, WavefrontCoordinate &selected,
    std::shared_ptr<PipelineState> &pipeline, bool &child_poison) noexcept {
  const auto fail = [](const Status status, const Check check,
                       const std::uint32_t stage) noexcept -> SelectResult {
    return SelectResult{status, check, stage};
  };
  for (std::size_t stage_index = 1u; stage_index < terminal_stage_;
       ++stage_index) {
    if (!wavefront_.already_ready(ticket.batch,
                                  static_cast<std::uint32_t>(stage_index))) {
      (void)ready_external_inputs(ticket, stage_index);
    }
  }

  bool projected = false;
  std::size_t supplied_inputs = 0u;
  while (!wavefront_.select(selected)) {
    if (wavefront_.promote(ticket.batch, selected)) {
      break;
    }
    bool progressed = false;
    const Status polled = prefetch_.poll(ticket, progressed);
    if (!polled) {
      return fail(polled, Check::PrefetchPoll, stage_id(ticket, selected));
    }
    if (progressed) {
      continue;
    }
    if (prefetch_.has_pending(ticket.batch)) {
      // Prefetcher::ready() is deliberately nonblocking. Yield while either
      // exact callback completes, then consume only the first ready receipt;
      // no ordinal wait can hide a later independent cell.
      std::this_thread::yield();
      continue;
    }
    {
      std::uint32_t resource = 0u;
      if (!wavefront_.forecast_middle(ticket.batch, selected, resource)) {
        return fail(Status::fail(Reason::PipelineInvalid), Check::Forecast,
                    stage_id(ticket, selected));
      }
      if (selected.batch != ticket.batch || selected.stage == 0u ||
          selected.stage >= terminal_stage_) {
        return fail(Status::fail(Reason::PipelineInvalid), Check::ForecastShape,
                    stage_id(ticket, selected));
      }
      if (!project_stage_scratch(graph_, run_, pool_, ticket, selected.stage,
                                 capacity_, scratch)) {
        return fail(Status::fail(Reason::PipelineInvalid),
                    Check::ForecastScratch, stage_id(ticket, selected));
      }
      projected = true;
      bool cleanup_failed = false;
      const Status supplied = prefetch_.supply_stage(ticket, scratch, selected,
                                                     resource, cleanup_failed);
      child_poison = cleanup_failed || child_poison;
      if (!supplied) {
        return fail(supplied, Check::SupplyStage, selected.stage);
      }
      ++supplied_inputs;
      if (supplied_inputs > residency::execution::GraphPromoteSourceCapacity) {
        return fail(Status::fail(Reason::PipelineInvalid),
                    Check::PromoteCapacity, selected.stage);
      }
    }
  }
  if (selected.batch != ticket.batch || selected.stage == 0u ||
      selected.stage >= terminal_stage_) {
    return fail(Status::fail(Reason::PipelineInvalid), Check::FinalShape,
                stage_id(ticket, selected));
  }
  const std::size_t stage_index = selected.stage;
  if (!projected && !project_stage_scratch(graph_, run_, pool_, ticket,
                                           stage_index, capacity_, scratch)) {
    return fail(Status::fail(Reason::PipelineInvalid), Check::FinalScratch,
                stage_id(ticket, selected));
  }
  const std::size_t pipeline_index =
      stage_index * residency::Pool::BankCount + ticket.bank;
  if (pipeline_index >= state_.graph_pipelines.size()) {
    return fail(Status::fail(Reason::PipelineInvalid), Check::PipelineIndex,
                stage_id(ticket, selected));
  }
  pipeline = graph_stage_pipeline(state_, stage_index, ticket.bank);
  if (pipeline == nullptr) {
    return fail(Status::fail(Reason::PipelineInvalid), Check::PipelineNull,
                stage_id(ticket, selected));
  }
  if (pipeline->residency_bank != ticket.bank) {
    return fail(Status::fail(Reason::PipelineInvalid), Check::PipelineBank,
                stage_id(ticket, selected));
  }
  if (pipeline->residency_graph_stage != stage_index) {
    return fail(Status::fail(Reason::PipelineInvalid), Check::PipelineStage,
                stage_id(ticket, selected));
  }
  return SelectResult{};
}

std::uint32_t
MiddleController::stage_id(const Ticket &ticket,
                           const WavefrontCoordinate &selected) const noexcept {
  if (selected.batch != ticket.batch || selected.stage == 0u ||
      selected.stage >= terminal_stage_) {
    return Fail::NoStage;
  }
  return selected.stage;
}

} // namespace rund::compute::detail::graph_reduce
