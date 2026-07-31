#pragma once

#include <kernel/core/model.hpp>

namespace rund::node::accel::detail::block {

inline constexpr rund::kernel::u64 MetalCompact = 1024u;
inline constexpr rund::kernel::u64 VulkanCompact = 256u;
inline constexpr rund::kernel::u64 MetalPartition = 1024u;
inline constexpr rund::kernel::u64 VulkanPartition = 256u;

} // namespace rund::node::accel::detail::block
