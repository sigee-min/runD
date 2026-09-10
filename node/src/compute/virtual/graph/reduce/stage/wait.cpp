#include "internal.hpp"

#include <rund/compute/pipeline/runtime.hpp>

namespace rund::compute::detail::graph_reduce {

Status StageController::wait(Ticket &ticket, const ExecutionStage stage,
                             const bool fold_now, bool &child_poison) noexcept {
  using ::rund::detail::counter::Accumulate;
  if (active_ != &ticket || ticket.submitted != stage) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::shared_ptr<PipelineState> &pipeline =
      stage == ExecutionStage::Prefix ? ticket.prefix : ticket.collective;
  if (pipeline == nullptr ||
      pipeline->residency_bank >= residency::Pool::BankCount) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const residency::ExecutionReceipt receipt =
      pool_.wait_execution(pipeline->residency_bank);
  active_ = nullptr;
  ticket.submitted = ExecutionStage::None;
  observe_timeline(stage == ExecutionStage::Prefix ? ticket.prefix_timeline
                                                   : ticket.collective_timeline,
                   receipt, stats_.pipeline.residency);
  Status folded = Status::success();
  if (fold_now || !receipt.status) {
    folded = fold(ticket, stage);
  }
  if (!receipt.status) {
    child_poison = !folded || poisoned_pipeline(pipeline) || child_poison;
    return folded ? receipt.status : folded;
  }
  if (!folded) {
    child_poison = true;
    return folded;
  }
  Accumulate(stats_.pipeline.residency.epoch_count, 1u);
  return Status::success();
}

} // namespace rund::compute::detail::graph_reduce
