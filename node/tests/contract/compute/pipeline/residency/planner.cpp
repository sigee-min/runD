#include "local.hpp"

#include "src/compute/pipeline/residency/planner.hpp"

#include <cstdint>

namespace rund_node_test_pipeline_residency {
namespace {

using namespace rund::compute::detail::residency;

[[nodiscard]] PlanResult plan(const std::uint64_t pages,
                              const std::uint64_t requested,
                              const std::uint64_t maximum) noexcept {
  return PlanResidency(StreamPlanInput{
      .page_bytes = 4096u,
      .page_count = pages,
      .requested_frames = requested,
      .max_frames = maximum,
  });
}

} // namespace

int CheckPlanner() {
  constexpr std::uint64_t pages = 1'000'000'000u;
  constexpr std::uint64_t frames = 4096u;
  auto planned = plan(pages, frames, frames);
  if (!planned || planned.plan.page_bytes() != 4096u ||
      !planned.plan.identity()) {
    return 1;
  }

  const StreamPlan &stream = planned.plan.stream();
  const std::uint64_t epochs = pages / frames + (pages % frames != 0u);
  PageRun first{};
  PageRun last{};
  if (stream.page_count() != pages || stream.frame_capacity() != frames ||
      stream.epoch_count() != epochs || !stream.epoch(0u, first) ||
      !stream.epoch(epochs - 1u, last) || first.first_page != 0u ||
      first.page_count != frames || last.first_page != (epochs - 1u) * frames ||
      last.page_count != pages - last.first_page ||
      stream.epoch(epochs, last)) {
    return 2;
  }

  const auto empty = plan(0u, 0u, 0u);
  PageRun none{};
  if (!empty || empty.plan.stream().page_count() != 0u ||
      empty.plan.stream().frame_capacity() != 0u ||
      empty.plan.stream().epoch_count() != 0u ||
      empty.plan.stream().epoch(0u, none)) {
    return 3;
  }
  if (plan(1u, 0u, 0u).failure != Failure::Infeasible ||
      plan(1u, 2u, 1u).failure != Failure::Capacity) {
    return 4;
  }
  const auto invalid = PlanResidency(StreamPlanInput{
      .page_count = 1u,
      .requested_frames = 1u,
      .max_frames = 1u,
  });
  return invalid.failure == Failure::Invalid ? 0 : 5;
}

} // namespace rund_node_test_pipeline_residency
