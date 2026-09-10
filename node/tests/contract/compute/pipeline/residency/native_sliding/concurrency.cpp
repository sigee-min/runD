#include "local.hpp"

#include <chrono>

namespace rund_node_test_pipeline_residency::native_sliding {

[[nodiscard]] bool SuppressionCase(const SuppressionStage stage) {
  SlidingFixture fixture{};
  if (!fixture.Prepare(2u)) {
    return false;
  }
  fixture.backends[0u]->delay_generation = 1u;
  fixture.backends[0u]->unknown_terminal = true;
  if (stage == SuppressionStage::Seed) {
    fixture.backends[1u]->block_seed = true;
  }
  fixture.StartWorkers();
  SlidingWait wait{};
  wait.coordinate_count = 2u;
  wait.stride = 2u;
  if (stage == SuppressionStage::Project) {
    wait.block_coordinate = 1u;
  }
  const auto request = MakeRequest(fixture, wait, 2u);
  bool submit_ok = false;
  std::thread submitter{[&] {
    submit_ok = accel::SubmitPreparedKernelPipelineSliding(
                    fixture.context, request, fixture.control)
                    .ok;
  }};
  bool entered = false;
  if (stage == SuppressionStage::Project) {
    std::unique_lock lock{wait.gate};
    entered = wait.ready.wait_for(lock, std::chrono::seconds{5},
                                  [&] { return wait.project_entered; });
  } else {
    std::unique_lock lock{fixture.backends[1u]->gate};
    entered = fixture.backends[1u]->ready.wait_for(
        lock, std::chrono::seconds{5},
        [&] { return fixture.backends[1u]->seed_entered; });
  }
  if (entered) {
    {
      std::lock_guard lock{fixture.backends[0u]->gate};
      fixture.backends[0u]->released = true;
    }
    fixture.backends[0u]->ready.notify_all();
    std::unique_lock lock{wait.gate};
    entered = wait.ready.wait_for(lock, std::chrono::seconds{5}, [&] {
      return wait.releases == 1u;
    }) && !wait.final;
  }
  if (stage == SuppressionStage::Project) {
    {
      std::lock_guard lock{wait.gate};
      wait.project_open = true;
    }
    wait.ready.notify_all();
  } else {
    {
      std::lock_guard lock{fixture.backends[1u]->gate};
      fixture.backends[1u]->seed_open = true;
    }
    fixture.backends[1u]->ready.notify_all();
  }
  submitter.join();
  return entered && submit_ok && WaitForFinal(wait) &&
         wait.evidence.terminal == accel::NativeTerminal::UnknownMayWrite &&
         wait.evidence.accepted_coordinates == 1u &&
         wait.evidence.released_coordinates == 1u &&
         wait.evidence.queue_calls == 1u &&
         fixture.backends[0u]->submission_count == 1u &&
         fixture.backends[1u]->submission_count == 0u;
}

[[nodiscard]] bool DelayedSubmitReturnCase() {
  SlidingFixture fixture{};
  if (!fixture.Prepare(2u)) {
    return false;
  }
  fixture.backends[0u]->delay_generation = 1u;
  fixture.backends[0u]->unknown_terminal = true;
  fixture.backends[1u]->block_submit_return = true;
  fixture.StartWorkers();
  SlidingWait wait{};
  wait.coordinate_count = 2u;
  wait.stride = 2u;
  const auto request = MakeRequest(fixture, wait, 2u);
  bool submit_ok = false;
  std::thread submitter{[&] {
    submit_ok = accel::SubmitPreparedKernelPipelineSliding(
                    fixture.context, request, fixture.control)
                    .ok;
  }};
  bool entered = false;
  {
    std::unique_lock lock{fixture.backends[1u]->gate};
    entered = fixture.backends[1u]->ready.wait_for(
        lock, std::chrono::seconds{5},
        [&] { return fixture.backends[1u]->submit_entered; });
  }
  if (entered) {
    {
      std::lock_guard lock{fixture.backends[0u]->gate};
      fixture.backends[0u]->released = true;
    }
    fixture.backends[0u]->ready.notify_all();
    std::unique_lock lock{wait.gate};
    entered = wait.ready.wait_for(lock, std::chrono::seconds{5}, [&] {
      return wait.releases == 1u;
    }) && !wait.final;
  }
  {
    std::lock_guard lock{fixture.backends[1u]->gate};
    fixture.backends[1u]->submit_open = true;
  }
  fixture.backends[1u]->ready.notify_all();
  submitter.join();
  return entered && submit_ok && WaitForFinal(wait) &&
         wait.evidence.terminal == accel::NativeTerminal::UnknownMayWrite &&
         wait.evidence.accepted_coordinates == 2u &&
         wait.evidence.released_coordinates == 2u &&
         wait.evidence.queue_calls == 2u &&
         fixture.backends[0u]->submission_count == 1u &&
         fixture.backends[1u]->submission_count == 1u;
}

[[nodiscard]] bool ReleaseHandoffCase() {
  SlidingFixture fixture{};
  if (!fixture.Prepare(2u)) {
    return false;
  }
  fixture.backends[1u]->delay_generation = 2u;
  fixture.backends[1u]->unknown_terminal = true;
  SlidingWait wait{};
  wait.coordinate_count = 2u;
  wait.stride = 2u;
  wait.block_release_coordinate = 0u;
  const auto request = MakeRequest(fixture, wait, 2u);
  if (!accel::SubmitPreparedKernelPipelineSliding(fixture.context, request,
                                                  fixture.control)
           .ok) {
    return false;
  }
  // Accept both slots before either terminal can arrive. An early completion
  // may deliver Release from Submit itself; blocking that callback would also
  // block the test thread that must open release_open. This case exercises
  // Release/Final handoff after the initial two-slot submission.
  fixture.StartWorkers();
  bool held = false;
  {
    std::unique_lock lock{wait.gate};
    held = wait.ready.wait_for(lock, std::chrono::seconds{5},
                               [&] { return wait.release_entered; });
  }
  if (held) {
    {
      std::lock_guard lock{fixture.backends[1u]->gate};
      fixture.backends[1u]->released = true;
    }
    fixture.backends[1u]->ready.notify_all();
    held = WaitForReleaseMask(wait, 3u) &&
           !wait.final_observed.load(std::memory_order_acquire);
  }
  {
    std::lock_guard lock{wait.gate};
    wait.release_open = true;
  }
  wait.ready.notify_all();
  return held && WaitForFinal(wait) &&
         wait.evidence.terminal == accel::NativeTerminal::UnknownMayWrite &&
         wait.evidence.accepted_coordinates == 2u &&
         wait.evidence.released_coordinates == 2u &&
         wait.evidence.queue_calls == 2u;
}

[[nodiscard]] bool ReturnedPressureCase() {
  SlidingFixture fixture{};
  if (!fixture.Prepare(2u)) {
    return false;
  }
  // Keep slot 1 submitted while slot 0 proves the explicit Wake edge.  If
  // both terminals race freely, slot 1 may legitimately re-arm and retire
  // slot 0 before the test thread calls Wake, making a correct completed run
  // look like a Wake rejection.
  fixture.backends[1u]->delay_generation = 2u;
  SlidingWait wait{};
  wait.coordinate_count = 2u;
  wait.stride = 2u;
  wait.block_returned_coordinate = 0u;
  wait.control = &fixture.control;
  const auto request = MakeRequest(fixture, wait, 2u);
  if (!accel::SubmitPreparedKernelPipelineSliding(fixture.context, request,
                                                  fixture.control)
           .ok) {
    return false;
  }
  // Starting the workers after the synchronous initial pump also excludes a
  // pre-existing terminal/service edge: slot 0's retained retry is now caused
  // by the explicit Wake under test, not by initial-submit overlap.
  fixture.StartWorkers();
  bool held = false;
  {
    std::unique_lock lock{wait.gate};
    held = wait.ready.wait_for(lock, std::chrono::seconds{5}, [&] {
      return wait.returned_entered;
    }) && !wait.final;
    wait.returned_open = true;
  }
  if (!held || !accel::WakePreparedKernelPipelineSliding(fixture.control).ok) {
    return false;
  }
  {
    std::lock_guard lock{fixture.backends[1u]->gate};
    fixture.backends[1u]->released = true;
  }
  fixture.backends[1u]->ready.notify_all();
  if (!WaitForFinal(wait)) {
    return false;
  }
  return wait.valid && wait.evidence.check.ok && wait.returned_count == 2u &&
         wait.evidence.accepted_coordinates == 2u &&
         wait.evidence.released_coordinates == 2u &&
         wait.evidence.queue_calls == 2u;
}

[[nodiscard]] bool ReturnedProgressRearmCase() {
  SlidingFixture fixture{};
  if (!fixture.Prepare(2u)) {
    return false;
  }
  fixture.backends[1u]->delay_generation = 2u;
  fixture.StartWorkers();
  SlidingWait wait{};
  wait.coordinate_count = 2u;
  wait.stride = 2u;
  wait.block_returned_coordinate = 0u;
  wait.returned_opens_on_peer = true;
  const auto request = MakeRequest(fixture, wait, 2u);
  if (!accel::SubmitPreparedKernelPipelineSliding(fixture.context, request,
                                                  fixture.control)
           .ok) {
    return false;
  }
  bool held = false;
  {
    std::unique_lock lock{wait.gate};
    held = wait.ready.wait_for(lock, std::chrono::seconds{5}, [&] {
      return wait.returned_entered;
    }) && !wait.final;
  }
  if (held) {
    {
      std::lock_guard lock{fixture.backends[1u]->gate};
      fixture.backends[1u]->released = true;
    }
    fixture.backends[1u]->ready.notify_all();
  }
  return held && WaitForFinal(wait) && wait.valid && wait.evidence.check.ok &&
         wait.returned_count == 2u &&
         wait.evidence.accepted_coordinates == 2u &&
         wait.evidence.released_coordinates == 2u &&
         wait.evidence.queue_calls == 2u;
}

[[nodiscard]] bool CrossSlotInlineCase() {
  SlidingFixture fixture{};
  if (!fixture.Prepare(2u)) {
    return false;
  }
  fixture.backends[0u]->delay_generation = 1u;
  fixture.backends[1u]->cross_completion_backend = fixture.backends[0u];
  fixture.StartWorkers();
  SlidingWait wait{};
  wait.coordinate_count = 2u;
  wait.stride = 2u;
  const auto request = MakeRequest(fixture, wait, 2u);
  return accel::SubmitPreparedKernelPipelineSliding(fixture.context, request,
                                                    fixture.control)
             .ok &&
         WaitForFinal(wait) && wait.valid && wait.evidence.check.ok &&
         wait.evidence.terminal == accel::NativeTerminal::Known &&
         wait.evidence.accepted_coordinates == 2u &&
         wait.evidence.released_coordinates == 2u;
}

[[nodiscard]] bool WrongControlGenerationCase() {
  SlidingFixture fixture{};
  if (!fixture.Prepare(1u)) {
    return false;
  }
  fixture.backends[0u]->generation_delta = 1;
  fixture.StartWorkers();
  SlidingWait wait{};
  wait.coordinate_count = 1u;
  wait.stride = 1u;
  const auto request = MakeRequest(fixture, wait, 1u);
  return accel::SubmitPreparedKernelPipelineSliding(fixture.context, request,
                                                    fixture.control)
             .ok &&
         WaitForFinal(wait) &&
         wait.evidence.terminal == accel::NativeTerminal::UnknownMayWrite &&
         !wait.evidence.check.ok && wait.evidence.accepted_coordinates == 1u &&
         wait.evidence.released_coordinates == 1u &&
         fixture.states[0u]->submission.quarantined;
}

} // namespace rund_node_test_pipeline_residency::native_sliding
