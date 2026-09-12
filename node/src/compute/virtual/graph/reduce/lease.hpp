#pragma once

#include "model.hpp"

namespace rund::compute::detail::graph_reduce {

[[nodiscard]] bool valid_stage_lease(const residency::EpochLease &,
                                     std::size_t pages,
                                     std::size_t ports) noexcept;
[[nodiscard]] residency::EpochLease prefix_input_lease(const Ticket &) noexcept;
[[nodiscard]] residency::EpochLease
collective_output_lease(const Ticket &) noexcept;

} // namespace rund::compute::detail::graph_reduce
