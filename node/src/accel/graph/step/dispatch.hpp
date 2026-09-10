#pragma once

#include "../step.hpp"

#include <cstdint>
#include <span>

namespace rund::node::accel::detail::step {

[[nodiscard]] FrozenDispatchCount
CountMapDispatch(const rund::kernel::ExecutionMetadata &metadata,
                 std::uint64_t element_count,
                 const rund::kernel::ComputeCaps &caps, std::uint64_t phase_id);

[[nodiscard]] FrozenDispatchCount
CountOriginalDispatch(std::span<const GraphCompileNode> nodes,
                      const rund::kernel::ComputeCaps &caps,
                      std::uint64_t phase_offset);

} // namespace rund::node::accel::detail::step
