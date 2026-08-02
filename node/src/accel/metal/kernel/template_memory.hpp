#pragma once

#include "../../kernel/memory.hpp"

namespace rund::node::accel::detail {

// Exact runD-owned host storage below one type-erased Metal template owner.
// Adapter-global source/library caches and opaque Metal objects are owned and
// observed by their respective cache/device authorities.
[[nodiscard]] PreparedMemory
ObserveMetalPipelineTemplate(const void *prepared) noexcept;

} // namespace rund::node::accel::detail
