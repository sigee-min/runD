#include "internal.hpp"

namespace rund::compute::detail::graph_reduce {

Status MiddleController::execute(Ticket &ticket, bool &child_poison) noexcept {
  const auto note = [&](const Status failure, const Check check,
                        const std::uint32_t stage = Fail::NoStage) noexcept {
    state_.failure_log.note(stage, ticket.batch, Phase::Middle, check, failure);
  };
  if (ticket.phase != TicketPhase::IntermediateDirty ||
      !ticket.intermediate_dirty || !stages_.idle()) {
    const Status failure = Status::fail(Reason::PipelineInvalid);
    note(failure, Check::Ticket);
    return failure;
  }
  for (std::size_t stage_index = 1u; stage_index < terminal_stage_;
       ++stage_index) {
    if (!wavefront_.already_ready(ticket.batch,
                                  static_cast<std::uint32_t>(stage_index))) {
      (void)ready_external_inputs(ticket, stage_index);
    }
  }
  StageScratch scratch{};
  for (std::size_t completed = 1u; completed < terminal_stage_; ++completed) {
    (void)completed;
    WavefrontCoordinate selected{};
    std::shared_ptr<PipelineState> pipeline;
    const SelectResult picked =
        select(ticket, scratch, selected, pipeline, child_poison);
    Status status = picked.status;
    if (status) {
      status = run_stage(ticket, scratch, selected, pipeline, child_poison);
    }
    if (!status) {
      note(status, picked.status ? Check::Execute : picked.check,
           picked.status ? stage_id(ticket, selected) : picked.stage);
      return status;
    }
  }
  const Status terminal = bind_terminal_input(ticket);
  if (!terminal) {
    const std::uint32_t stage =
        terminal_stage_ < graph_.stages().size() &&
                terminal_stage_ <= std::numeric_limits<std::uint32_t>::max()
            ? static_cast<std::uint32_t>(terminal_stage_)
            : Fail::NoStage;
    note(terminal, Check::TerminalBind, stage);
  }
  return terminal;
}

} // namespace rund::compute::detail::graph_reduce
