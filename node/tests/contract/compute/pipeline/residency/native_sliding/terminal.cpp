#include "local.hpp"

namespace rund_node_test_pipeline_residency::native_sliding {

[[nodiscard]] bool ProjectionFailureCase(const bool accepted_sibling) {
  SlidingFixture fixture{};
  const std::size_t roles = accepted_sibling ? 2u : 1u;
  if (!fixture.Prepare(roles)) {
    return false;
  }
  SlidingWait wait{};
  wait.coordinate_count = accepted_sibling ? 4u : 1u;
  wait.stride = roles;
  wait.fail_coordinate = accepted_sibling ? 1u : 0u;
  if (accepted_sibling) {
    fixture.backends[0u]->delay_generation = 1u;
    fixture.StartWorkers();
  }
  const auto request = MakeRequest(fixture, wait, wait.coordinate_count);
  if (!accel::SubmitPreparedKernelPipelineSliding(fixture.context, request,
                                                  fixture.control)
           .ok) {
    return false;
  }
  if (accepted_sibling) {
    {
      std::lock_guard lock{wait.gate};
      if (wait.final) {
        return false;
      }
    }
    {
      std::lock_guard lock{fixture.backends[0u]->gate};
      fixture.backends[0u]->released = true;
    }
    fixture.backends[0u]->ready.notify_all();
  }
  if (!WaitForFinal(wait)) {
    return false;
  }
  const std::uint64_t expected = accepted_sibling ? 1u : 0u;
  return wait.evidence.terminal == accel::NativeTerminal::Known &&
         !wait.evidence.check.ok &&
         wait.evidence.first_failure_coordinate == wait.fail_coordinate &&
         wait.evidence.accepted_coordinates == expected &&
         wait.evidence.released_coordinates == expected &&
         wait.evidence.queue_calls == expected &&
         fixture.backends[0u]->submission_count == expected &&
         (!accepted_sibling || fixture.backends[1u]->submission_count == 0u);
}

[[nodiscard]] bool InlineContradictionCase() {
  SlidingFixture fixture{};
  if (!fixture.Prepare(1u)) {
    return false;
  }
  fixture.backends[0u]->inline_completion = true;
  fixture.backends[0u]->reject_after_inline = true;
  SlidingWait wait{};
  wait.coordinate_count = 1u;
  wait.stride = 1u;
  const auto request = MakeRequest(fixture, wait, 1u);
  if (!accel::SubmitPreparedKernelPipelineSliding(fixture.context, request,
                                                  fixture.control)
           .ok ||
      !WaitForFinal(wait)) {
    return false;
  }
  return wait.evidence.terminal == accel::NativeTerminal::UnknownMayWrite &&
         !wait.evidence.check.ok && wait.evidence.accepted_coordinates == 1u &&
         wait.evidence.released_coordinates == 1u &&
         wait.evidence.queue_calls == 1u &&
         fixture.states[0u]->submission.quarantined;
}

[[nodiscard]] bool CrossThreadEarlyContradictionCase() {
  SlidingFixture fixture{};
  if (!fixture.Prepare(1u)) {
    return false;
  }
  fixture.backends[0u]->cross_thread_early_completion = true;
  fixture.backends[0u]->reject_after_cross_thread = true;
  SlidingWait wait{};
  wait.coordinate_count = 1u;
  wait.stride = 1u;
  const auto request = MakeRequest(fixture, wait, 1u);
  if (!accel::SubmitPreparedKernelPipelineSliding(fixture.context, request,
                                                  fixture.control)
           .ok ||
      !WaitForFinal(wait)) {
    return false;
  }
  bool completion_returned = false;
  {
    std::lock_guard lock{fixture.backends[0u]->gate};
    completion_returned = fixture.backends[0u]->early_completion_returned;
  }
  return completion_returned &&
         wait.evidence.terminal == accel::NativeTerminal::UnknownMayWrite &&
         !wait.evidence.check.ok && wait.evidence.accepted_coordinates == 1u &&
         wait.evidence.released_coordinates == 1u &&
         wait.evidence.queue_calls == 1u &&
         fixture.states[0u]->submission.quarantined;
}

[[nodiscard]] bool ActiveSelfRetainCase() {
  SlidingFixture fixture{};
  if (!fixture.Prepare(1u)) {
    return false;
  }
  fixture.backends[0u]->delay_generation = 1u;
  fixture.StartWorkers();
  SlidingWait wait{};
  wait.coordinate_count = 1u;
  wait.stride = 1u;
  const auto request = MakeRequest(fixture, wait, 1u);
  if (!accel::SubmitPreparedKernelPipelineSliding(fixture.context, request,
                                                  fixture.control)
           .ok) {
    return false;
  }
  fixture.control.state.reset();
  {
    std::lock_guard lock{fixture.backends[0u]->gate};
    fixture.backends[0u]->released = true;
  }
  fixture.backends[0u]->ready.notify_all();
  return WaitForFinal(wait) && wait.evidence.check.ok &&
         wait.evidence.terminal == accel::NativeTerminal::Known &&
         wait.evidence.accepted_coordinates == 1u &&
         wait.evidence.released_coordinates == 1u;
}

} // namespace rund_node_test_pipeline_residency::native_sliding
