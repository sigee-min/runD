#pragma once

#include "epoch.hpp"

namespace rund::compute::detail {

// Executes a complete non-Scan stream through the two canonical physical
// banks. One Device-global Authority owns every state transition; the worker
// executes only an already pinned bank and never selects or remaps a frame.
[[nodiscard]] VirtualEpochResult
execute_virtual_cpu_overlap(VirtualPipelineState &state, VirtualBacking &input,
                            VirtualBacking &output,
                            const VirtualRunProjection &run, Stats &stats,
                            ::rund::node::hash_detail::Fnv &output_hash,
                            VirtualReduction *reduction) noexcept;

[[nodiscard]] VirtualEpochResult
execute_virtual_accel_overlap(VirtualPipelineState &state,
                              VirtualBacking &input, VirtualBacking &output,
                              const VirtualRunProjection &run, Stats &stats,
                              ::rund::node::hash_detail::Fnv &output_hash,
                              VirtualReduction *reduction) noexcept;

} // namespace rund::compute::detail
