#include "local.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/accel/kernel/prepared.hpp"
#include "src/compute/pipeline/local.hpp"
#include "src/compute/pipeline/state.hpp"

#include <memory>
#include <utility>

namespace rund_node_test_pipeline {

[[nodiscard]] int
CheckUnknownCompletionProfileIdentity(rund::compute::Device &device,
                                      const Backend backend) {
#if defined(RUND_NODE_TEST_BACKEND_CPU)
  static_cast<void>(device);
  static_cast<void>(backend);
  return 0;
#else
  using namespace rund::compute;
  if (backend == Backend::Cpu) {
    return 0;
  }
  constexpr std::array<std::int32_t, 4u> input{1, 2, 3, 4};
  auto program =
      on(device)
          .map<std::int32_t>("pipeline-unknown-completion", input.size(),
                             [](auto value) { return value + 1; })
          .compile();
  auto source = Upload(device, input);
  auto output = device.buffer<std::int32_t>(input.size());
  if (!program || !source || !output) {
    return 1;
  }
  auto prepared = pipeline(device)
                      .profile(PipelineProfile::Steps)
                      .then(*program, read(*source), write(*output))
                      .prepare();
  if (!prepared) {
    return 2;
  }
  const std::shared_ptr<detail::PipelineState> &state =
      detail::PipelineStateAccess::state(*prepared);
  if (state == nullptr || state->active_step_count != 1u ||
      !detail::queue_pipeline(state)) {
    return 3;
  }

  std::array<rund::node::accel::detail::PreparedPipelineStepEvidence, 1u>
      native_rows{{
          {.original_dispatch_count = 1u,
           .final_dispatch_count = 1u,
           .physical_dispatch_count = 1u,
           .workgroup_count = 7u,
           .work_item_count = 13u,
           .duration_ns = 17u,
           .timing_sample_count = 1u,
           .work_sample_count = 1u,
           .clock =
               rund::node::accel::detail::PreparedPipelineStepClock::Device,
           .relation = rund::node::accel::detail::
               PreparedPipelineStepTimingRelation::NonAdditive},
      }};
  rund::node::accel::detail::PreparedPipelineEvidence unknown{
      .shared = {.backend = backend == Backend::Metal ? rund::AccelApi::Metal
                                                      : rund::AccelApi::Vulkan,
                 .command_submit_count = 1u,
                 .ok = false,
                 .reason = "compute_backend_failed"},
      .check = {false, "compute_backend_failed"},
      .control = {.generation =
                      static_cast<std::uint32_t>(state->attempt_generation),
                  .reason = static_cast<std::uint32_t>(Reason::Ok),
                  .failed_step =
                      rund::node::accel::detail::PreparedPipelineNoStep,
                  .verified_prefix = 1u},
      .profile = {.steps = native_rows,
                  .instrumentation_command_count = 5u,
                  .instrumentation_byte_count = 64u,
                  .observed = true},
      .control_byte_count =
          rund::node::accel::detail::PreparedPipelineControlBytes,
      .active_step_count = 1u,
      .submitted = true,
      .control_observed = true,
      .control_valid = false,
  };
  const Status completed =
      detail::finish_pipeline_on(state, std::move(unknown));
  std::array<PipelineStepProfile, 1u> rows{};
  const auto profile = prepared->profile(rows);
  if (completed || completed.reason() != Reason::CompletionInvalid ||
      !profile || profile->execution.pipeline.verified_step_count != 0u ||
      profile->execution.pipeline.failed_step_index !=
          PipelineStats::no_failed_step ||
      rows[0].execution.available() || rows[0].timing.available() ||
      profile->instrumentation_command_count != 5u ||
      profile->instrumentation_byte_count != 64u ||
      !ProfileMemoryReconciles(*profile, rows)) {
    return 4;
  }
  return 0;
#endif
}

} // namespace rund_node_test_pipeline
