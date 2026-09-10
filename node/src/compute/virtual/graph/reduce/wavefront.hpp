#pragma once

#include "../../../pipeline/residency/model.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace rund::compute::detail {
struct VirtualPipelineState;
struct VirtualRunProjection;
} // namespace rund::compute::detail

namespace rund::compute::detail::graph_reduce {

[[nodiscard]] bool
graph_wavefront_pair_eligible(const VirtualPipelineState &,
                              const VirtualRunProjection &) noexcept;

inline constexpr std::size_t WavefrontBankCount = 2u;
inline constexpr std::size_t WavefrontStageCapacity =
    residency::TiledGraphPortCapacity / 2u;
inline constexpr std::size_t WavefrontCellCapacity =
    WavefrontBankCount * WavefrontStageCapacity;

struct WavefrontCoordinate final {
  std::uint64_t ordinal{};
  std::uint64_t batch{};
  std::uint64_t first_page{};
  std::uint64_t page_count{};
  std::uint32_t stage{};
  std::uint32_t resource{};

  [[nodiscard]] constexpr bool
  operator==(const WavefrontCoordinate &) const noexcept = default;
};

// Allocation-free ready-node controller for the two execution banks retained
// by the Graph product route. It consumes only planner-sealed predecessors;
// callers separately prove HostReady and the exact H2D/device-resident handoff.
class Wavefront final {
public:
  [[nodiscard]] bool reset(const residency::TiledGraphInvocation &,
                           const residency::TiledGraphPlan &) noexcept;
  [[nodiscard]] bool admit(std::uint64_t batch) noexcept;

  // A successful typed Forecast terminal opens Host readiness only. It never
  // impersonates physical Promote. device_ready() is called after the existing
  // exact Authority lease has made all external inputs device-visible.
  [[nodiscard]] bool forecast_terminal(std::uint64_t batch, std::uint32_t stage,
                                       std::uint32_t resource, Status) noexcept;
  // Synchronous CPU backing service has no asynchronous Forecast object; its
  // exact returned service fact enters through the same HostReady boundary.
  [[nodiscard]] bool host_ready(std::uint64_t batch, std::uint32_t stage,
                                std::uint32_t resource) noexcept;
  [[nodiscard]] bool device_ready(std::uint64_t batch,
                                  std::uint32_t stage) noexcept;
  // Authority proved that every required external binding is already resident
  // in the selected Device bank, so no Host Forecast/transfer exists.
  [[nodiscard]] bool device_resident(std::uint64_t batch,
                                     std::uint32_t stage) noexcept;
  [[nodiscard]] bool already_ready(std::uint64_t batch,
                                   std::uint32_t stage) const noexcept;
  // A stage with only transient inputs has no physical Host service boundary;
  // its planner-sealed predecessor terminals are its sole readiness facts.
  [[nodiscard]] bool dependency_ready(std::uint64_t batch,
                                      std::uint32_t stage) noexcept;

  // When no DeviceReady cell can run, the controller may advance one exact
  // external-input dependency without reverting to ordinal stage policy.
  // forecast() returns the least planner-ready cell/resource still missing a
  // Host receipt. promote() returns the least requested-batch middle cell
  // whose complete HostReady mask may now be joined to an exact destination
  // lease.
  [[nodiscard]] bool forecast(WavefrontCoordinate &,
                              std::uint32_t &resource) const noexcept;
  // Middle execution may forecast only a nonterminal cell. The public
  // planner contract still exposes forecast() for lower-level tests that
  // inspect every stage, so this bounded view is kept explicit here. The
  // caller's batch is part of the admission boundary.
  [[nodiscard]] bool forecast_middle(std::uint64_t, WavefrontCoordinate &,
                                     std::uint32_t &resource) const noexcept;
  // Pure pair admission probe. It returns the exact two current-batch
  // independent middle cells without reserving either Forecast bit.
  [[nodiscard]] bool
  pair_forecast(std::uint64_t batch, std::array<WavefrontCoordinate, 2u> &,
                std::array<std::uint32_t, 2u> &) const noexcept;
  // Reserve one exact missing resource before submitting its Forecast. The
  // reservation is a planner fact only: it does not make the cell HostReady
  // and is cleared by Forecast terminal or explicit rollback.
  [[nodiscard]] bool reserve_forecast(const WavefrontCoordinate &,
                                      std::uint32_t resource) noexcept;
  [[nodiscard]] bool cancel_forecast(const WavefrontCoordinate &,
                                     std::uint32_t resource) noexcept;
  [[nodiscard]] bool promote(std::uint64_t,
                             WavefrontCoordinate &) const noexcept;

  [[nodiscard]] bool select(WavefrontCoordinate &) const noexcept;
  [[nodiscard]] bool dispatch(const WavefrontCoordinate &) noexcept;
  [[nodiscard]] bool cancel_dispatch(const WavefrontCoordinate &) noexcept;
  [[nodiscard]] bool terminal(const WavefrontCoordinate &) noexcept;
  [[nodiscard]] bool terminal(std::uint64_t batch,
                              std::uint32_t stage) noexcept;
  [[nodiscard]] bool release(std::uint64_t batch) noexcept;
  // Failure-only reset after all Authority/native/Forecast owners have been
  // terminally cleaned. No readiness fact survives into a retry.
  void abort() noexcept;

  [[nodiscard]] constexpr std::size_t retained_cells() const noexcept {
    return cells_.size();
  }

private:
  enum class CellState : std::uint8_t { Empty, Waiting, Running, Terminal };

  struct Cell final {
    WavefrontCoordinate coordinate{};
    std::array<residency::TiledGraphDependency,
               residency::TiledGraphDependencyCapacity>
        predecessors{};
    std::array<std::uint32_t, residency::TiledGraphResourceCapacity>
        external_inputs{};
    std::uint32_t required_mask{};
    std::uint32_t host_ready_mask{};
    std::uint32_t forecast_mask{};
    std::size_t predecessor_count{};
    std::size_t external_input_count{};
    CellState state{CellState::Empty};
    bool device_ready{};
    bool failed{};
  };

  [[nodiscard]] Cell *find(std::uint64_t, std::uint32_t) noexcept;
  [[nodiscard]] const Cell *find(std::uint64_t, std::uint32_t) const noexcept;
  [[nodiscard]] bool predecessors_ready(const Cell &) const noexcept;
  [[nodiscard]] static bool before(const WavefrontCoordinate &,
                                   const WavefrontCoordinate &) noexcept;

  const residency::TiledGraphInvocation *invocation_{};
  const residency::TiledGraphPlan *plan_{};
  std::array<Cell, WavefrontCellCapacity> cells_{};
  std::array<std::array<WavefrontCoordinate, WavefrontStageCapacity>,
             WavefrontBankCount>
      terminal_history_{};
  std::array<std::array<bool, WavefrontStageCapacity>, WavefrontBankCount>
      terminal_valid_{};
  std::array<std::uint64_t, WavefrontBankCount> release_history_{};
  std::array<bool, WavefrontBankCount> release_valid_{};
  std::size_t stage_count_{};
};

} // namespace rund::compute::detail::graph_reduce
