#include "local.hpp"

#include "src/compute/pipeline/residency/planner.hpp"

#include <cstdint>

namespace rund_node_test_pipeline_residency {
namespace {

using namespace rund::compute::detail::residency;

[[nodiscard]] PlanResult
plan(const std::uint64_t pages, const std::uint64_t requested,
     const std::uint64_t maximum,
     const std::uint64_t prefetch_distance = 2u) noexcept {
  return PlanResidency(StreamPlanInput{
      .page_bytes = 4096u,
      .page_count = pages,
      .requested_frames = requested,
      .max_frames = maximum,
      .dirty = DirtyRange{.offset = 3072u, .bytes = 1024u},
      .dirty_bytes = pages == 0u ? 0u : pages * 1024u - 512u,
      .prefetch_distance = prefetch_distance,
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
  DirtyRange first_dirty{};
  DirtyRange last_dirty{};
  std::uint64_t first_use = 0u;
  std::uint64_t last_use = 0u;
  std::uint64_t first_next = 0u;
  std::uint64_t last_next = 0u;
  if (stream.page_count() != pages || stream.frame_capacity() != frames ||
      stream.dirty_extent() != DirtyRange{.offset = 3072u, .bytes = 1024u} ||
      stream.dirty_bytes() != pages * 1024u - 512u ||
      stream.prefetch_distance() != 2u || stream.epoch_count() != epochs ||
      !stream.epoch(0u, first) || !stream.epoch(epochs - 1u, last) ||
      first.first_page != 0u || first.page_count != frames ||
      first.pin != PinInterval{0u, 0u} || first.prefetch_epoch != 0u ||
      first.ready_epoch != 0u || last.first_page != (epochs - 1u) * frames ||
      last.page_count != pages - last.first_page ||
      last.pin != PinInterval{epochs - 1u, epochs - 1u} ||
      last.prefetch_epoch != epochs - 3u || last.ready_epoch != epochs - 1u ||
      !stream.dirty_extent(0u, first_dirty) ||
      first_dirty != DirtyRange{.offset = 3072u, .bytes = 1024u} ||
      !stream.dirty_extent(pages - 1u, last_dirty) ||
      last_dirty != DirtyRange{.offset = 3072u, .bytes = 512u} ||
      stream.dirty_extent(pages, last_dirty) ||
      !stream.first_use(0u, first_use) || first_use != 0u ||
      !stream.first_use(pages - 1u, last_use) || last_use != pages - 1u ||
      stream.first_use(pages, last_use) || !stream.next_use(0u, first_next) ||
      first_next != pages || !stream.next_use(pages - 1u, last_next) ||
      last_next != pages * 2u - 1u || stream.next_use(pages, last_next) ||
      stream.epoch(epochs, last)) {
    return 2;
  }

  const auto empty = plan(0u, 0u, 0u);
  PageRun none{};
  if (!empty || empty.plan.stream().page_count() != 0u ||
      empty.plan.stream().frame_capacity() != 0u ||
      empty.plan.stream().epoch_count() != 0u ||
      empty.plan.stream().epoch(0u, none) ||
      empty.plan.stream().first_use(0u, first_use) ||
      empty.plan.stream().next_use(0u, first_next)) {
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
      .dirty = DirtyRange{.bytes = 1u},
  });
  const auto invalid_dirty = PlanResidency(StreamPlanInput{
      .page_bytes = 4096u,
      .page_count = 1u,
      .requested_frames = 1u,
      .max_frames = 1u,
      .dirty = DirtyRange{.offset = 4095u, .bytes = 2u},
      .dirty_bytes = 2u,
  });
  const auto invalid_dirty_count = PlanResidency(StreamPlanInput{
      .page_bytes = 4096u,
      .page_count = 2u,
      .requested_frames = 1u,
      .max_frames = 1u,
      .dirty = DirtyRange{.offset = 3072u, .bytes = 1024u},
      .dirty_bytes = 1024u,
  });
  return invalid.failure == Failure::Invalid &&
                 invalid_dirty.failure == Failure::Invalid &&
                 invalid_dirty_count.failure == Failure::Invalid
             ? 0
             : 5;
}

} // namespace rund_node_test_pipeline_residency
