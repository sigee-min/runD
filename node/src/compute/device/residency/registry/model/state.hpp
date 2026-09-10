#pragma once

#include "cpu.hpp"
#include "cycle.hpp"
#include "execution.hpp"
#include "view.hpp"

#include "../credentials/cpu.hpp"
#include "../view_receipt.hpp"

#include "../../execution/graph_forecast.hpp"
#include "../../execution/graph_persist/capacity.hpp"

#include <array>
#include <cstdint>
#include <memory>

namespace rund::compute::detail::graph_reduce {
struct CpuGraphQuarantine;
} // namespace rund::compute::detail::graph_reduce

namespace rund::compute::detail::residency::registry_model {

// Each component owns one mutable authority domain.  The enclosing Authority
// supplies the sole gate and frame table; these records only group state that
// already shared one lifecycle and do not mirror any other registry.
struct ExecutionAuthorityState final {
  ExecutionSlot slot{};
};

struct CycleAuthorityState final {
  std::array<LeaseSlot, 4u> epochs{};
  LeaseSlot writeback{};
  std::array<LeaseSlot, execution::GraphPersistSlotCapacity>
      graph_persists{};
  CycleSlot cycle{};
  std::uint64_t retired_cycle{};
};

struct CpuGraphAuthorityState final {
  std::shared_ptr<graph_reduce::CpuGraphQuarantine> cpu_quarantine{};
  std::array<execution::GraphForecast, GraphForecastQuarantineCapacity>
      graph_forecast_quarantine{};
  CpuReservationKey pending_cpu{};
};

struct ViewAuthorityState final {
  std::unique_ptr<ViewCommitReceipt> view_quarantine{};
  std::unique_ptr<ViewCommitReceipt> view_idle{};
  std::uint64_t active_view_commit_stamp{};
  std::uint64_t next_view_stamp{1u};
};

struct CredentialsState final {
  std::uint64_t owner_id{};
  std::uint64_t next_token{1u};
  std::uint64_t next_generation{1u};
};

} // namespace rund::compute::detail::residency::registry_model
