#pragma once

#include "../cycle/model.hpp"

#include <cstdint>

namespace rund::compute::detail::residency {

class Authority;

// Stateless owner for the production rolling compute-flight journal.  The
// Authority remains the sole owner of the gate, frame table, epoch slots, and
// CycleSlot; this facet owns only the cycle protocol algorithms.
class CycleOwner final {
public:
  explicit CycleOwner(Authority &) noexcept;

  CycleOwner(const CycleOwner &) noexcept = default;
  CycleOwner &operator=(const CycleOwner &) = delete;

  [[nodiscard]] bool bind_cycle(const cycle::Flight &, std::uint64_t &) noexcept;
  [[nodiscard]] bool advance_cycle(std::uint64_t,
                                   const cycle::Flight &) noexcept;
  [[nodiscard]] bool complete_cycle(std::uint64_t, std::uint64_t, bool,
                                    bool invalidate_all = false) noexcept;
  [[nodiscard]] bool close_cycle(std::uint64_t) noexcept;

private:
  Authority &authority_;
};

} // namespace rund::compute::detail::residency
