#include "internal.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU) && \
    !defined(RUND_NODE_TEST_BACKEND_METAL) && \
    defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace rund_node_test_pipeline_vulkan_residency {

[[nodiscard]] int CheckSchedule(ResidencyFixture &fixture) {
  // A Schedule is one whole-run native recurrence, not a sequence of W4
  // submissions. Four logical roles encode bank x publication parity; Q is
  // recovered arithmetically from role=e%4 and turn=e/4. Q=17 crosses four
  // complete role turns and a tail while retaining only the four roles.
  ScheduleWait schedule{};
  schedule.context = fixture.wait.context;
  schedule.pipelines = fixture.wait.pipelines;
  schedule.first_generation = {41'000u, 51'000u, 41'001u, 51'001u};
  schedule.generation_stride = 2u;
  schedule.plan = 0x56'4b'53u;
  schedule.token = 61'001u;
  schedule.generation = 71'001u;
  schedule.epoch_count = 17u;
  accel::PreparedResidencyScheduleRequest schedule_request{
      .plan_identity = schedule.plan,
      .token = schedule.token,
      .generation = schedule.generation,
      .epoch_count = schedule.epoch_count,
      .role_count = 4u,
      .tail_local_count = 1u,
      .release = CompleteScheduleRelease,
      .final = CompleteScheduleFinal,
      .user = &schedule,
  };
  for (std::size_t role = 0u; role < schedule_request.role_count; ++role) {
    schedule_request.roles[role].pipeline = schedule.pipelines[role % 2u];
    schedule_request.roles[role].locals[0u] = 0u;
    schedule_request.roles[role].local_count = 1u;
    schedule_request.roles[role].first_control_generation =
        schedule.first_generation[role];
    schedule_request.roles[role].control_generation_stride =
        schedule.generation_stride;
    schedule_request.roles[role].role = static_cast<std::uint8_t>(role);
    schedule_request.roles[role].bank = static_cast<std::uint8_t>(role % 2u);
  }
  const accel::BackendResidencySchedulePreparation schedule_capability =
      accel::PrepareKernelPipelineSchedule(
          schedule.context, schedule_request.roles,
          schedule_request.epoch_count, schedule_request.tail_local_count);
  schedule_request.lowering = schedule_capability.owner;
  accel::PreparedResidencyStreamControl schedule_stream{};
  accel::PreparedResidencyScheduleControl schedule_control{};
  if (!schedule_capability.check.ok ||
      schedule_capability.kind !=
          accel::BackendResidencyScheduleLowering::VulkanTimeline ||
      schedule_capability.queue_calls != 1u ||
      !schedule_capability.callbacks_async ||
      schedule_capability.transient_bytes == 0u ||
      !accel::ClaimPreparedKernelPipelineStream(
           schedule.context, schedule.pipelines, schedule.plan, schedule.token,
           schedule.generation, schedule_stream)
           .ok ||
      !accel::SubmitPreparedKernelPipelineSchedule(
           schedule.context, schedule_request, schedule_control,
           schedule_stream)
           .ok ||
      schedule.done.load(std::memory_order_acquire) ||
      schedule.releases.load(std::memory_order_acquire) != 0u) {
    return 12;
  }
  for (const std::uint64_t epoch : {1u, 0u}) {
    if (!accel::SignalPreparedKernelPipelineSchedule(
             schedule.context, schedule.pipelines[epoch % 2u],
             accel::BackendResidencyWindowSignal{
                 .plan_identity = schedule.plan,
                 .token = schedule.token,
                 .generation = schedule.generation,
                 .epoch = epoch,
                 .control_generation =
                     ScheduleControlGeneration(schedule, epoch),
                 .bank = static_cast<std::uint8_t>(epoch % 2u),
             })
             .ok) {
      return 12;
    }
  }
  schedule.done.wait(false, std::memory_order_acquire);
  if (!schedule.valid.load(std::memory_order_acquire) ||
      schedule.releases.load(std::memory_order_acquire) !=
          schedule.epoch_count ||
      !schedule.final.check.ok ||
      schedule.final.terminal != accel::NativeTerminal::Known ||
      schedule.final.public_handoffs != 1u ||
      schedule.final.native_batches != schedule.epoch_count ||
      schedule.final.queue_calls != 1u ||
      schedule.final.native_inflight_peak != 2u ||
      schedule.final.released_prefix != schedule.epoch_count ||
      schedule.final.completed_prefix != schedule.epoch_count ||
      schedule.final.suppressed_count != 0u || schedule_control.active ||
      schedule_control.quarantined || !schedule_stream.active ||
      schedule_stream.quarantined ||
      !accel::ReleasePreparedKernelPipelineStream(
           schedule_stream, schedule.plan, schedule.token, schedule.generation,
           false)
           .ok) {
    return 13;
  }
  return 0;
}

} // namespace rund_node_test_pipeline_vulkan_residency

#endif
