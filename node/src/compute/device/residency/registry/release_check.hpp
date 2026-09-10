#pragma once

#include "../registry.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace rund::compute::detail::residency::release_detail {

// Declaration-only seam for the Authority release invariant. Each validation
// phase has one compiled implementation owner under registry/release_check/.
struct ReleaseCheck final {
  [[nodiscard]] static bool row(const Authority &,
                                std::span<const FrameRegion>, std::uint32_t,
                                bool &) noexcept;
  [[nodiscard]] static bool
  direct_binding(const Authority &,
                 const ResidentRecurrenceBinding &) noexcept;
  [[nodiscard]] static bool region(const Authority &,
                                   std::span<const FrameRegion>, FrameRegion,
                                   bool &) noexcept;
  [[nodiscard]] static bool idle(const Authority::LeaseSlot &) noexcept;
  [[nodiscard]] static bool active(const Authority::LeaseSlot &) noexcept;

  [[nodiscard]] static bool retry(const Authority &,
                                  const Authority::LeaseSlot &) noexcept;
  [[nodiscard]] static bool persist(const Authority &,
                                    const Authority::LeaseSlot &) noexcept;
  [[nodiscard]] static bool graph_rows(const Authority &) noexcept;
  [[nodiscard]] static bool slot(const Authority &,
                                 std::span<const FrameRegion>,
                                 const Authority::LeaseSlot &) noexcept;

  [[nodiscard]] static bool
  execution(const Authority &, std::span<const FrameRegion>,
            const registry_model::ExecutionSlot &) noexcept;

  [[nodiscard]] static bool cycle(const Authority &,
                                  std::span<const FrameRegion>,
                                  const registry_model::CycleSlot &) noexcept;
  [[nodiscard]] static bool non_epoch_cycle_clear(const Authority &) noexcept;
  [[nodiscard]] static bool allowed(const Authority &,
                                    std::span<const FrameRegion>) noexcept;
};

} // namespace rund::compute::detail::residency::release_detail
