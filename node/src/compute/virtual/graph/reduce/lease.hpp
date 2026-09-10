#pragma once

#include "model.hpp"

namespace rund::compute::detail::graph_reduce {

[[nodiscard]] bool
retain_prefix_lease(Ticket &, const residency::AuthorityResult &) noexcept;
[[nodiscard]] bool
retain_collective_lease(Ticket &, const residency::AuthorityResult &) noexcept;
[[nodiscard]] residency::EpochLease prefix_lease(Ticket &) noexcept;
[[nodiscard]] residency::EpochLease prefix_input_lease(Ticket &) noexcept;
[[nodiscard]] residency::EpochLease collective_lease(Ticket &) noexcept;
[[nodiscard]] residency::EpochLease collective_output_lease(Ticket &) noexcept;

} // namespace rund::compute::detail::graph_reduce
