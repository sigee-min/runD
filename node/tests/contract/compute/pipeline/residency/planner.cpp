#include "local.hpp"

#include "src/compute/pipeline/residency/planner.hpp"

#include <cstdint>

namespace rund_node_test_pipeline_residency {
namespace {

using namespace rund::compute::detail::residency;

[[nodiscard]] PlanResult plan(const std::uint64_t pages,
                              const std::uint64_t requested,
                              const std::uint64_t maximum) noexcept {
  return PlanResidency(PlanInput{
      .page_bytes = 4096u,
      .page_count = pages,
      .requested_slots = requested,
      .max_slots = maximum,
  });
}

} // namespace

int CheckPlanner() {
  constexpr std::uint64_t pages = 1'000'000'000u;
  constexpr std::uint64_t slots = 4096u;
  auto planned = plan(pages, slots, slots);
  if (!planned || planned.plan.page_bytes() != 4096u ||
      !planned.plan.identity()) {
    return 1;
  }

  const LinearPlan &linear = planned.plan.linear();
  const std::uint64_t waves = pages / slots + (pages % slots != 0u);
  PageRun first{};
  PageRun last{};
  if (linear.page_count() != pages || linear.slot_capacity() != slots ||
      linear.wave_count() != waves || !linear.wave(0u, first) ||
      !linear.wave(waves - 1u, last) || first.first_page != 0u ||
      first.page_count != slots || last.first_page != (waves - 1u) * slots ||
      last.page_count != pages - last.first_page || linear.wave(waves, last)) {
    return 2;
  }

  const auto empty = plan(0u, 0u, 0u);
  PageRun none{};
  if (!empty || empty.plan.linear().page_count() != 0u ||
      empty.plan.linear().slot_capacity() != 0u ||
      empty.plan.linear().wave_count() != 0u ||
      empty.plan.linear().wave(0u, none)) {
    return 3;
  }
  if (plan(1u, 0u, 0u).failure != Failure::Infeasible ||
      plan(1u, 2u, 1u).failure != Failure::Capacity) {
    return 4;
  }
  const auto invalid = PlanResidency(PlanInput{
      .page_count = 1u,
      .requested_slots = 1u,
      .max_slots = 1u,
  });
  return invalid.failure == Failure::Invalid ? 0 : 5;
}

} // namespace rund_node_test_pipeline_residency
