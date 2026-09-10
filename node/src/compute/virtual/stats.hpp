#pragma once

#include <rund/compute/stats.hpp>
#include <rund/compute/status.hpp>

namespace rund::compute::detail {

[[nodiscard]] Status accumulate_virtual_epoch(Stats &total,
                                              const Stats &epoch) noexcept;

// Graph stages are independently compiled physical Programs, so their
// backend graph hashes must differ. This accumulator preserves every actual
// producer counter while publishing the caller's single canonical graph/VSM
// identity instead of treating the stage hashes as competing authorities.
[[nodiscard]] Status
accumulate_virtual_graph_stage(Stats &total, const Stats &stage,
                               std::uint64_t graph_identity) noexcept;

} // namespace rund::compute::detail
