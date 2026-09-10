#include "internal.hpp"

#include <rund/compute/pipeline/runtime.hpp>

namespace rund::compute::detail::graph_reduce {

void StageController::abort_active(bool &child_poison) noexcept {
  using ::rund::detail::counter::Accumulate;
  if (active_ == nullptr) {
    return;
  }
  Ticket &ticket = *active_;
  const ExecutionStage stage = ticket.submitted;
  const std::shared_ptr<PipelineState> &pipeline =
      stage == ExecutionStage::Prefix ? ticket.prefix : ticket.collective;
  const residency::ExecutionReceipt receipt =
      pipeline == nullptr
          ? residency::ExecutionReceipt{.status = Status::fail(
                                            Reason::PipelineInvalid)}
          : pool_.wait_execution(pipeline->residency_bank);
  active_ = nullptr;
  ticket.submitted = ExecutionStage::None;
  observe_timeline(stage == ExecutionStage::Prefix ? ticket.prefix_timeline
                                                   : ticket.collective_timeline,
                   receipt, stats_.pipeline.residency);
  const Status folded = fold(ticket, stage);
  if (receipt.status) {
    Accumulate(stats_.pipeline.residency.epoch_count, 1u);
  } else {
    child_poison = poisoned_pipeline(pipeline) || child_poison;
  }
  child_poison = !folded || child_poison;
}

} // namespace rund::compute::detail::graph_reduce
