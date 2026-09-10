#pragma once

#include "model.hpp"

namespace rund::compute::detail::residency::cycle {

// Seals one immutable, bounded temporal DAG. This function owns ordering only:
// Authority admission must already have assigned each nonzero phase token, and
// a backend adapter must consume the returned nodes without choosing frames,
// victims, or publication order.
[[nodiscard]] Result seal(std::span<const Epoch> epochs) noexcept;

} // namespace rund::compute::detail::residency::cycle
