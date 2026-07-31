#pragma once

#include "../backend/catalog/entry.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] BackendEntry VulkanEntry() noexcept;

} // namespace rund::node::accel::detail
