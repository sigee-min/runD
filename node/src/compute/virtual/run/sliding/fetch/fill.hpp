#pragma once

#include "../../../../device/residency/execution/model.hpp"

#include <cstddef>

namespace rund::compute::detail::sliding_product_detail {

// Completes only bytes outside the authenticated backing slice. The recipe is
// already sealed into FetchSource; no Window semantic is reconstructed here.
[[nodiscard]] bool
fill_fetch_frame(std::byte *,
                 const residency::execution::FetchSource &) noexcept;

} // namespace rund::compute::detail::sliding_product_detail
