#pragma once

#include "model.hpp"

#include <rund/compute/stats.hpp>

#include <cstdint>

namespace rund_node_test_virtual::product {

// final_run is one physical terminal execution. Backing facts are the
// independent cold-plus-warm cohort totals; they are never synthesized by
// multiplying final-run telemetry.
struct ProductExecutionEvidence final {
  rund::compute::Stats final_run{};
  BackingFacts input_cohort{};
  BackingFacts output_cohort{};
  std::uint64_t warm_host_allocations{};
  std::uint64_t observed_hash{};
  bool same_capacity{};
  bool tail_poisoned{};
  bool profile_matches{};
};

[[nodiscard]] bool
ProductExecutionMatches(const ProductExecutionEvidence &evidence) noexcept;

} // namespace rund_node_test_virtual::product
