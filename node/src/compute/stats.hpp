#pragma once

#include <rund/compute/stats.hpp>

namespace rund {
struct AccelEvidence;
struct AccelRunFacts;
} // namespace rund

namespace rund::compute::detail {

[[nodiscard]] Stats
stats_from_evidence(Backend backend, const rund::AccelEvidence &evidence,
                    std::uint64_t graph_read_bytes) noexcept;

// Sole semantic mapper from physical backend facts into public Stats. Runtime
// completion and cold prepared-stream publication both call this owner.
void accumulate_run_facts(Stats &stats,
                          const rund::AccelRunFacts &facts) noexcept;

} // namespace rund::compute::detail
