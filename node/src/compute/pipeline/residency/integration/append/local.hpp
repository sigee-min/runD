#pragma once

#include "../../integration.hpp"

namespace rund::compute::detail {

[[nodiscard]] PipelineBinding
append_frame_binding(std::uint32_t owner, Type type, FixedFormat format,
                     std::size_t count, std::size_t offset,
                     std::size_t backing_count, ResourceAccess access) noexcept;

} // namespace rund::compute::detail
