#include "model.hpp"

#include <memory>
#include <utility>

namespace rund::compute::detail::graph_reduce {

bool reset_ticket(Ticket &ticket) noexcept {
  if (ticket.host_ready_count != 0u || ticket.input_promote ||
      ticket.output_drain || ticket.output_persist ||
      ticket.prefix_token != 0u || ticket.collective_token != 0u ||
      ticket.submitted != ExecutionStage::None ||
      ticket.prefix_receipt.occupied() ||
      ticket.collective_receipt.occupied() ||
      ticket.supply_receipt.occupied() || ticket.middle_receipt.occupied()) {
    return false;
  }
  CpuEpochReceipt prefix = std::move(ticket.prefix_receipt);
  CpuEpochReceipt collective = std::move(ticket.collective_receipt);
  CpuEpochReceipt supply = std::move(ticket.supply_receipt);
  CpuEpochReceipt middle = std::move(ticket.middle_receipt);
  std::destroy_at(&ticket);
  std::construct_at(&ticket);
  ticket.prefix_receipt = std::move(prefix);
  ticket.collective_receipt = std::move(collective);
  ticket.supply_receipt = std::move(supply);
  ticket.middle_receipt = std::move(middle);
  return true;
}

} // namespace rund::compute::detail::graph_reduce
