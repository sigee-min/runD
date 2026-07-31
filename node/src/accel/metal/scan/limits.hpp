#pragma once

#include <cstdint>

namespace rund::node::accel::detail {

inline constexpr std::uint64_t kMetalScanMaxBlockSize = 1024u;
inline constexpr std::uint64_t kMetalScanWidth = 128u;

} // namespace rund::node::accel::detail
