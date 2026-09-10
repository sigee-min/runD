#include "local.hpp"

#include <chrono>
#include <cstring>
#include <utility>

namespace rund_node_test_pipeline_residency::native_sliding {

void CompleteReentrantFinal(
    void *const raw,
    accel::BackendResidencySlidingFinal &&final) noexcept {
  auto *const wait = static_cast<ReentrantWait *>(raw);
  if (wait == nullptr || wait->fixture == nullptr) {
    return;
  }
  bool resubmit = false;
  {
    std::lock_guard lock{wait->gate};
    if (wait->finals == 0u) {
      wait->first = std::move(final);
      wait->coordinate_count = 5u;
      wait->releases = 0u;
      wait->slot_turn.fill(0u);
      wait->slot_last.fill(std::numeric_limits<std::uint64_t>::max());
      resubmit = true;
    } else {
      wait->evidence = std::move(final);
      wait->final = true;
    }
    ++wait->finals;
  }
  if (resubmit) {
    auto request = MakeRequest(*wait->fixture, *wait, 5u, 111u, 222u, 333u);
    request.final = CompleteReentrantFinal;
    const bool ok = accel::SubmitPreparedKernelPipelineSliding(
                        wait->fixture->context, request, wait->fixture->control)
                        .ok;
    {
      std::lock_guard lock{wait->gate};
      wait->resubmit_ok = ok;
      wait->resubmit_returned = true;
      if (!ok) {
        wait->final = true;
      }
    }
  }
  wait->ready.notify_all();
}

[[nodiscard]] bool ReentrantFinalCase() {
  SlidingFixture fixture{};
  if (!fixture.Prepare()) {
    return false;
  }
  fixture.StartWorkers();
  ReentrantWait wait{};
  wait.fixture = &fixture;
  wait.coordinate_count = 1u;
  wait.stride = fixture.role_count;
  auto request = MakeRequest(fixture, wait, 1u);
  request.final = CompleteReentrantFinal;
  if (!accel::SubmitPreparedKernelPipelineSliding(fixture.context, request,
                                                  fixture.control)
           .ok) {
    return false;
  }
  {
    std::unique_lock lock{wait.gate};
    if (!wait.ready.wait_for(lock, std::chrono::seconds{5}, [&] {
          return wait.final && wait.resubmit_returned;
        })) {
      return false;
    }
  }
  return wait.resubmit_ok && wait.finals == 2u && wait.valid &&
         wait.first.check.ok &&
         wait.first.terminal == accel::NativeTerminal::Known &&
         wait.first.coordinate_count == 1u &&
         wait.first.accepted_coordinates == 1u &&
         wait.first.released_coordinates == 1u && wait.evidence.check.ok &&
         wait.evidence.terminal == accel::NativeTerminal::Known &&
         wait.evidence.coordinate_count == 5u &&
         wait.evidence.accepted_coordinates == 5u &&
         wait.evidence.released_coordinates == 5u &&
         wait.evidence.queue_calls == 5u;
}

void SubmitNested(void *const raw) noexcept {
  auto *const nested = static_cast<NestedSubmit *>(raw);
  if (nested == nullptr || nested->fixture == nullptr ||
      nested->wait == nullptr) {
    return;
  }
  const auto request = MakeRequest(*nested->fixture, *nested->wait, 1u);
  nested->ok = accel::SubmitPreparedKernelPipelineSliding(
                   nested->fixture->context, request, nested->fixture->control)
                   .ok;
  nested->returned = true;
}

[[nodiscard]] bool NestedInlineCase() {
  SlidingFixture inner{};
  SlidingFixture outer{};
  if (!inner.Prepare(1u) || !outer.Prepare(1u)) {
    return false;
  }
  inner.backends[0u]->inline_completion = true;
  outer.backends[0u]->inline_completion = true;
  SlidingWait inner_wait{};
  inner_wait.coordinate_count = 1u;
  inner_wait.stride = 1u;
  SlidingWait outer_wait{};
  outer_wait.coordinate_count = 1u;
  outer_wait.stride = 1u;
  NestedSubmit nested{.fixture = &inner, .wait = &inner_wait};
  outer.backends[0u]->nested_submit = SubmitNested;
  outer.backends[0u]->nested_user = &nested;
  const auto request = MakeRequest(outer, outer_wait, 1u);
  return accel::SubmitPreparedKernelPipelineSliding(outer.context, request,
                                                    outer.control)
             .ok &&
         nested.returned && nested.ok && WaitForFinal(inner_wait) &&
         WaitForFinal(outer_wait) &&
         inner_wait.evidence.terminal ==
             accel::NativeTerminal::UnknownMayWrite &&
         outer_wait.evidence.terminal ==
             accel::NativeTerminal::UnknownMayWrite &&
         inner_wait.evidence.accepted_coordinates == 1u &&
         outer_wait.evidence.accepted_coordinates == 1u;
}

[[nodiscard]] bool ReverseFailureCase() {
  SlidingFixture fixture{};
  if (!fixture.Prepare(2u)) {
    return false;
  }
  fixture.backends[0u]->delay_generation = 1u;
  fixture.backends[1u]->delay_generation = 2u;
  fixture.backends[0u]->known_failure = true;
  fixture.backends[1u]->known_failure = true;
  fixture.backends[0u]->failure_reason = "compute_backend_failed";
  fixture.backends[1u]->failure_reason = "compute_device_busy";
  fixture.StartWorkers();
  SlidingWait wait{};
  wait.coordinate_count = 2u;
  wait.stride = 2u;
  const auto request = MakeRequest(fixture, wait, 2u);
  if (!accel::SubmitPreparedKernelPipelineSliding(fixture.context, request,
                                                  fixture.control)
           .ok) {
    return false;
  }
  {
    std::lock_guard lock{fixture.backends[1u]->gate};
    fixture.backends[1u]->released = true;
  }
  fixture.backends[1u]->ready.notify_all();
  {
    std::unique_lock lock{wait.gate};
    if (!wait.ready.wait_for(lock, std::chrono::seconds{5},
                             [&] { return wait.releases == 1u; }) ||
        wait.final) {
      return false;
    }
  }
  {
    std::lock_guard lock{fixture.backends[0u]->gate};
    fixture.backends[0u]->released = true;
  }
  fixture.backends[0u]->ready.notify_all();
  return WaitForFinal(wait) &&
         wait.evidence.terminal == accel::NativeTerminal::Known &&
         !wait.evidence.check.ok &&
         wait.evidence.first_failure_coordinate == 0u &&
         std::strcmp(wait.evidence.check.reason, "compute_backend_failed") ==
             0;
}

[[nodiscard]] bool MultiReentrantWakeCase() {
  SlidingFixture fixture{};
  if (!fixture.Prepare(1u)) {
    return false;
  }
  fixture.StartWorkers();
  SlidingWait wait{};
  wait.coordinate_count = 1u;
  wait.stride = 1u;
  wait.control = &fixture.control;
  wait.reentrant_wake = true;
  wait.reentrant_wake_limit = 4u;
  wait.pending_remaining = 3u;
  const auto request = MakeRequest(fixture, wait, 1u);
  return accel::SubmitPreparedKernelPipelineSliding(fixture.context, request,
                                                    fixture.control)
             .ok &&
         WaitForFinal(wait) && wait.valid && wait.evidence.check.ok &&
         wait.reentrant_wake_count == 4u &&
         wait.evidence.accepted_coordinates == 1u &&
         wait.evidence.released_coordinates == 1u;
}

} // namespace rund_node_test_pipeline_residency::native_sliding
