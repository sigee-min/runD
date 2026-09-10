#pragma once

#include "../run.hpp"

#include "../model.hpp"
#include "../../recurrence/plan.hpp"
#include "reservation.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>

namespace rund::node::accel::detail {

struct PreparedKernelTemplateRegistryState;
class PipelineBudgetTransaction;

[[nodiscard]] rund::AccelCheck reserve_pipeline_budget(
    PreparedKernelTemplateRegistry &registry,
    std::span<const PreparedKernelRun *const> runs,
    std::span<const BackendRecurrence> recurrences,
    const PreparedKernelPipelineReservation &structure,
    const PreparedMapRecurrenceReservation &map_recurrence,
    std::uint32_t route_copies,
    PipelineBudgetTransaction &transaction) noexcept;

class PipelineBudgetTransaction final {
public:
  PipelineBudgetTransaction() noexcept;
  ~PipelineBudgetTransaction();

  PipelineBudgetTransaction(const PipelineBudgetTransaction &) = delete;
  PipelineBudgetTransaction &
  operator=(const PipelineBudgetTransaction &) = delete;

  void commit() noexcept;

private:
  friend rund::AccelCheck reserve_pipeline_budget(
      PreparedKernelTemplateRegistry &registry,
      std::span<const PreparedKernelRun *const> runs,
      std::span<const BackendRecurrence> recurrences,
      const PreparedKernelPipelineReservation &structure,
      const PreparedMapRecurrenceReservation &map_recurrence,
      std::uint32_t route_copies,
      PipelineBudgetTransaction &transaction) noexcept;

  [[nodiscard]] bool begin(PreparedKernelTemplateRegistryState &state,
                           PreparedKernelTemplateRegistry &registry) noexcept;
  void rollback() noexcept;

  PreparedKernelTemplateRegistryState *state_{};
  PreparedKernelTemplateRegistry *registry_{};
  std::unique_lock<std::recursive_mutex> lock_{};
  std::size_t entry_count_{};
  std::size_t charge_count_{};
  PreparedKernelPipelineReservation consumed_{};
  PreparedKernelPipelineReservation reservation_{};
  bool committed_{};
};

// Returns the exact retained host footprint reserved for the registry's
// capacity. The private entry/charge sizes remain owned by registry/state.cpp.
[[nodiscard]] bool PreparedKernelTemplateRegistryBytes(
    std::uint64_t template_count, std::uint64_t &bytes) noexcept;

} // namespace rund::node::accel::detail
