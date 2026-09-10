#pragma once

#include "../internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

// Copies one Authority-sealed canonical Window footprint into the K expanded
// Device input targets. Geometry and boundary recipes come only from the
// move-only Promote capability.
[[nodiscard]] Status
assemble_window_frames(SlidingProductRun &, std::uint64_t,
                       const residency::execution::SlidingPromote &) noexcept;

} // namespace rund::compute::detail::sliding_product_detail
