#include "../../registry/execution_owner.hpp"

#include "../../execution/plan.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>

namespace rund::compute::detail::residency {

bool ExecutionOwner::issue_execution(const std::uint64_t token,
                                     const std::uint64_t generation,
                                     const execution::Plan &plan,
                                     const execution::Node &node,
                                     const std::uint64_t sequence,
                                     ExecutionTicket &ticket) noexcept {
  ticket = {};
  std::lock_guard lock{authority_.gate_};
  std::size_t bank = 0u;
  std::size_t phase = 0u;
  if (!validate_execution_issue_locked(token, generation, plan, node, sequence,
                                       bank, phase)) {
    return false;
  }
  const bool window_service =
      authority_.execution_state_.slot.window_cache_admitted &&
      node.domain == execution::Domain::HostService;
  if (window_service &&
      !issue_execution_window_service_locked(plan, node, bank)) {
    return false;
  }
  if (!issue_execution_output_service_locked(node)) {
    return false;
  }
  finalize_execution_issue_locked(token, generation, plan, node, sequence, bank,
                                  phase, window_service, ticket);
  return true;
}

} // namespace rund::compute::detail::residency
