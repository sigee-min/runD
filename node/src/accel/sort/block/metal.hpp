#pragma once

#include "bucket.hpp"

namespace rund::node::accel::detail {

inline constexpr rund::kernel::u32 kMetalSortThreadCount = 256u;
// Eight fixed rounds amortize histogram initialization, block-prefix traffic,
// and threadgroup publication while retaining 256-lane occupancy. The
// pipeline cache name includes this value, so changing the physical shape can
// never reuse an incompatible compiled kernel.
inline constexpr rund::kernel::u32 kMetalSortRounds = 8u;
inline constexpr rund::kernel::u32 kMetalSortSimdWidth = 32u;
inline constexpr rund::kernel::u32 kMetalSortSimdGroupCount =
    kMetalSortThreadCount / kMetalSortSimdWidth;
inline constexpr rund::kernel::u32 kMetalSortPackedPairs =
    kMetalSortSimdGroupCount / 2u;
inline constexpr rund::kernel::u32 kMetalSortBlockSize =
    kMetalSortThreadCount * kMetalSortRounds;

} // namespace rund::node::accel::detail
