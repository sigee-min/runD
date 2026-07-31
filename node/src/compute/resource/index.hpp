#pragma once

#include <rund/compute/resource/plan.hpp>

#include <cstdint>
#include <span>

namespace rund::compute::resource::detail {

// Private, test-visible operation counts. These are not product telemetry;
// they make the interval index complexity contract verifiable without timing.
struct AnalysisStats final {
  std::uint64_t insert_visits{};
  std::uint64_t query_visits{};
  std::uint64_t envelope_candidates{};
  std::uint64_t exact_checks{};
};

[[nodiscard]] Result<Plan> analyze_measured(std::span<const Resource> resources,
                                            std::span<const Access> accesses,
                                            std::uint32_t node_count,
                                            AnalysisStats &stats);

} // namespace rund::compute::resource::detail
