#pragma once

#include "../pool.hpp"

namespace rund::compute::detail::residency {

[[nodiscard]] FrameRole
frame_role(GraphResourceRole role) noexcept;

[[nodiscard]] bool
graph_classes_match(const Pool &pool,
                    std::span<const TiledGraphPhysicalClass> classes) noexcept;

} // namespace rund::compute::detail::residency
