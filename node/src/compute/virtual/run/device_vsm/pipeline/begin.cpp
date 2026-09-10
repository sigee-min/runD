#include "../internal.hpp"

namespace rund::compute::detail::device_vsm_product_detail {

Status begin_pipelines(DeviceVsmProductRun &run) noexcept {
  if (run.owner == nullptr || run.projection == nullptr ||
      run.projection->frame_capacity == 0u ||
      run.projection->frame_capacity > residency::execution::UseCapacity) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::array<std::uint32_t, residency::execution::UseCapacity> locals{};
  const std::size_t count = run.projection->frame_capacity;
  for (std::size_t local = 0u; local < count; ++local) {
    locals[local] = static_cast<std::uint32_t>(local);
  }
  for (std::size_t bank = 0u; bank < run.owner->pipeline_count; ++bank) {
    PipelineState &pipeline = *run.owner->pipelines[bank];
    const PipelineClaimAuthority authority =
        run.projection->poolless_device_vsm()
            ? PipelineClaimAuthority::Shared
            : PipelineClaimAuthority::PrivateResidency;
    Status started = Status::success();
    {
      std::lock_guard lock{pipeline.gate};
      started = start_pipeline(pipeline, authority);
      if (started && authority == PipelineClaimAuthority::PrivateResidency) {
        const PipelineOutcome prepared = prepare_residency_pipeline_execution(
            pipeline, std::span<const std::uint32_t>{locals.data(), count});
        started = prepared.status;
        if (!started && pipeline.phase == PipelinePhase::Running) {
          publish_pipeline_terminal(
              pipeline,
              PipelineTerminal{.reason = started.reason(),
                               .publication_suppressed = true},
              authority);
        }
      }
    }
    if (!started) {
      reject_pipelines(run, started);
      return started;
    }
    run.pipeline_started[bank] = true;
  }
  return Status::success();
}

void reject_pipelines(DeviceVsmProductRun &run, const Status failure) noexcept {
  if (run.owner == nullptr) {
    return;
  }
  for (std::size_t bank = 0u; bank < run.owner->pipeline_count; ++bank) {
    PipelineState &pipeline = *run.owner->pipelines[bank];
    std::lock_guard lock{pipeline.gate};
    if (run.pipeline_started[bank] &&
        pipeline.phase == PipelinePhase::Running) {
      publish_pipeline_terminal(
          pipeline,
          PipelineTerminal{.reason = failure.reason(),
                           .writes_possible = false,
                           .publication_suppressed = true},
          run.projection != nullptr && run.projection->poolless_device_vsm()
              ? PipelineClaimAuthority::Shared
              : PipelineClaimAuthority::PrivateResidency);
    }
    run.pipeline_started[bank] = false;
  }
}

} // namespace rund::compute::detail::device_vsm_product_detail
