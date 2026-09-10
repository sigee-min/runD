#pragma once

#include "../model.hpp"

#include <cstdint>

namespace rund::node::accel::detail {

// Checked reservation accumulation is shared by structure projection, budget
// accounting, and registry transactions. Keep this arithmetic in the one
// reservation owner so every caller has identical overflow behavior.
[[nodiscard]] bool accumulate(std::uint64_t &target,
                              std::uint64_t value) noexcept;

[[nodiscard]] bool accumulate_reservation(
    PreparedKernelPipelineReservation &target,
    const PreparedKernelPipelineReservation &value) noexcept;

} // namespace rund::node::accel::detail
