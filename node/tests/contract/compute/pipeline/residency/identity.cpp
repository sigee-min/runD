#include "local.hpp"

#include "src/compute/pipeline/residency/planner.hpp"

#include <array>
#include <cstdint>

namespace rund_node_test_pipeline_residency {
namespace {

using namespace rund::compute::detail::residency;

[[nodiscard]] PlanResult
plan(const std::uint64_t page_bytes, const std::uint64_t pages,
     const std::uint64_t requested, const std::uint64_t maximum = 4u,
     const DirtyRange dirty = {3072u, 1024u},
     const std::uint64_t prefetch_distance = 2u,
     const std::uint64_t dirty_tail_trim = 0u) noexcept {
  return PlanResidency(StreamPlanInput{
      .page_bytes = page_bytes,
      .page_count = pages,
      .requested_frames = requested,
      .max_frames = maximum,
      .dirty = dirty,
      .dirty_bytes = pages == 0u ? 0u : pages * dirty.bytes - dirty_tail_trim,
      .prefetch_distance = prefetch_distance,
  });
}

} // namespace

int CheckIdentity() {
  const auto canonical = plan(4096u, 3u, 2u);
  const auto same = plan(4096u, 3u, 2u);
  const auto same_with_larger_limit = plan(4096u, 3u, 2u, 8u);
  const auto saturated = plan(4096u, 3u, 3u);
  const auto saturated_by_request = plan(4096u, 3u, 4u);
  if (!canonical || !same || !same_with_larger_limit || !saturated ||
      !saturated_by_request || !canonical.plan.identity() ||
      canonical.plan.identity() != same.plan.identity() ||
      canonical.plan.identity() != same_with_larger_limit.plan.identity() ||
      saturated.plan.identity() != saturated_by_request.plan.identity()) {
    return 1;
  }

  const std::array changed{
      plan(8192u, 3u, 2u),
      plan(4096u, 4u, 2u),
      plan(4096u, 3u, 1u),
      plan(4096u, 3u, 2u, 4u, DirtyRange{2048u, 2048u}),
      plan(4096u, 3u, 2u, 4u, DirtyRange{3072u, 1024u}, 1u),
      plan(4096u, 3u, 2u, 4u, DirtyRange{3072u, 1024u}, 2u, 1u),
  };
  for (const PlanResult &candidate : changed) {
    if (!candidate || candidate.plan.identity() == canonical.plan.identity()) {
      return 2;
    }
  }
  return 0;
}

} // namespace rund_node_test_pipeline_residency
