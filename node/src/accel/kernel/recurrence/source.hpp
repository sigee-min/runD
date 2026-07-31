#pragma once

#include "../recurrence.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] bool TransformSource(rund::kernel::LoweringArtifact &artifact,
                                   std::uint64_t input_count,
                                   std::uint64_t output_count);

} // namespace rund::node::accel::detail
