#pragma once

#include <rund/compute/stats.hpp>

namespace rund {
struct AccelEvidence;
}

namespace rund::compute::detail {

[[nodiscard]] Stats
stats_from_evidence(Backend backend,
                    const rund::AccelEvidence &evidence,
                    std::uint64_t graph_read_bytes) noexcept;

} // namespace rund::compute::detail
