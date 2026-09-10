#include "../persist.hpp"

namespace rund::compute::detail::graph_reduce {

Status PersistController::start(Ticket &ticket, Timeline *const hidden_by,
                                bool &child_poison) noexcept {
  if (output_ == nullptr || !ticket.output_dirty || ticket.intermediate_dirty ||
      owner_ == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const Status prepared = prepare_host_output(ticket, hidden_by, child_poison);
  if (!prepared) {
    return prepared;
  }
  const Status issued = output_persist_detail::issue(
      authority_, owner_, run_, graph_, terminal_stage_, ticket,
      ticket.output_persist);
  if (!issued) {
    return issued;
  }
  return ticket.collective->device->backend == Backend::Cpu
             ? persist_cpu(ticket, child_poison)
             : submit_async(ticket, child_poison);
}

} // namespace rund::compute::detail::graph_reduce
