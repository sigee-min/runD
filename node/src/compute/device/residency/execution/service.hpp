#pragma once

#include "../registry/credentials/execution.hpp"
#include "plan.hpp"

namespace rund::compute::detail::residency::execution {

struct Release;

// Fixed two-bank Host-service/native-release join. Authority remains the sole
// node journal; Service owns only the monotonic issue coordinate used by the
// bounded Window coordinator.
class Service final {
public:
  [[nodiscard]] bool bind(Authority &, const Plan &,
                          const ExecutionLease &) noexcept;
  [[nodiscard]] bool issue(std::uint64_t epoch, Phase,
                           ExecutionTicket &) noexcept;
  [[nodiscard]] bool terminal(const ExecutionTicket &, ExecutionTerminal,
                              Status) noexcept;
  [[nodiscard]] bool release(const Release &) noexcept;

private:
  Authority *authority_{};
  const Plan *plan_{};
  ExecutionLease lease_{};
  std::uint64_t sequence_{1u};
};

} // namespace rund::compute::detail::residency::execution
