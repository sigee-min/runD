#include "internal.hpp"

namespace rund::compute::detail::graph_reduce {

Status StageController::fold(Ticket &ticket,
                             const ExecutionStage stage) noexcept {
  bool &executed = stage == ExecutionStage::Prefix ? ticket.prefix_executed
                                                   : ticket.collective_executed;
  bool &folded = stage == ExecutionStage::Prefix
                     ? ticket.prefix_stats_folded
                     : ticket.collective_stats_folded;
  const std::shared_ptr<PipelineState> &pipeline =
      stage == ExecutionStage::Prefix ? ticket.prefix : ticket.collective;
  if (!executed || folded || pipeline == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  folded = true;
  return fold_stage(stats_, pipeline, identity_);
}

void StageController::fold_pending(const std::span<Ticket> tickets,
                                   bool &child_poison) noexcept {
  for (Ticket &ticket : tickets) {
    if (ticket.prefix_executed && !ticket.prefix_stats_folded) {
      child_poison = !fold(ticket, ExecutionStage::Prefix) || child_poison;
    }
    if (ticket.collective_executed && !ticket.collective_stats_folded) {
      child_poison = !fold(ticket, ExecutionStage::Collective) || child_poison;
    }
  }
}

} // namespace rund::compute::detail::graph_reduce
