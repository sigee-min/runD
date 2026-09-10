#pragma once

#include "../internal.hpp"

namespace rund::compute::detail::sliding_product_detail {

struct FetchBackingSlice final {
  std::uint64_t offset{};
  std::uint64_t target_offset{};
  std::uint64_t bytes{};
};

// Applies only the Authority-minted resident overlap and returns the exact
// remaining backing slice. It does not read backing or fill boundary bytes.
[[nodiscard]] bool
prepare_fetch_backing_slice(const SlidingProductRun &,
                            const residency::execution::SlidingFetch &,
                            std::byte *, FetchBackingSlice &) noexcept;

} // namespace rund::compute::detail::sliding_product_detail
