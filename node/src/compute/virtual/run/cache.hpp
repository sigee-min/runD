#pragma once

#include "../../device/residency.hpp"
#include "projection.hpp"

#include <rund/compute/stats.hpp>

#include <span>

namespace rund::compute {
class VirtualBacking;
}

namespace rund::compute::detail {
struct PipelineState;

[[nodiscard]] Status supply_residency_cache(PipelineState &pipeline,
                                            const VirtualRunProjection &run,
                                            residency::EpochLease lease,
                                            bool scan, Stats &stats) noexcept;

[[nodiscard]] Status retain_residency_output(PipelineState &pipeline,
                                             const VirtualRunProjection &run,
                                             residency::EpochLease lease,
                                             Stats &stats) noexcept;

[[nodiscard]] Status writeback_residency_cache(
    VirtualBacking &output, const VirtualRunProjection &run,
    std::span<const residency::CacheTransition> transitions,
    ResidencyStats &residency_stats) noexcept;

} // namespace rund::compute::detail
