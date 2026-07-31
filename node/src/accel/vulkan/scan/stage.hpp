#pragma once

#include <cstdint>

namespace rund::node::accel::detail {

enum class VulkanScanStage : std::uint8_t { Block, Prefix, Offset };

} // namespace rund::node::accel::detail
