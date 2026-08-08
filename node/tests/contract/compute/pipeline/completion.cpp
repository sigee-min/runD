#include "local.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/accel/kernel/prepared.hpp"
#include "src/compute/pipeline/local.hpp"
#include "src/compute/pipeline/state.hpp"

#include <memory>
#include <type_traits>
#include <utility>

namespace rund_node_test_pipeline {
namespace {

using rund::compute::Status;
using rund::compute::detail::CpuPipelineSelection;
using rund::compute::detail::CpuPipelineSelectionDisposition;
using rund::compute::detail::JobState;

static_assert(!std::is_aggregate_v<CpuPipelineSelection>);
static_assert(!std::is_default_constructible_v<CpuPipelineSelection>);
static_assert(!std::is_copy_constructible_v<CpuPipelineSelection>);
static_assert(std::is_nothrow_move_constructible_v<CpuPipelineSelection>);
static_assert(!std::is_move_assignable_v<CpuPipelineSelection>);

[[nodiscard]] int CheckCpuPipelineSelectionContract() {
  const CpuPipelineSelection failed =
      CpuPipelineSelection::failed(Status::success());
  if (failed.disposition() != CpuPipelineSelectionDisposition::Failed ||
      failed.status().reason() != rund::compute::Reason::PipelineInvalid ||
      failed.job() != nullptr || failed.step() != 0u) {
    return 1;
  }

  const CpuPipelineSelection exact_failure = CpuPipelineSelection::failed(
      Status::fail(rund::compute::Reason::BoundedCountInvalid));
  if (exact_failure.disposition() != CpuPipelineSelectionDisposition::Failed ||
      exact_failure.status().reason() !=
          rund::compute::Reason::BoundedCountInvalid ||
      exact_failure.job() != nullptr || exact_failure.step() != 0u) {
    return 2;
  }

  const CpuPipelineSelection null_job = CpuPipelineSelection::selected({}, 17u);
  if (null_job.disposition() != CpuPipelineSelectionDisposition::Failed ||
      null_job.status().reason() != rund::compute::Reason::PipelineInvalid ||
      null_job.job() != nullptr || null_job.step() != 0u) {
    return 3;
  }

  const std::shared_ptr<JobState> job = std::make_shared<JobState>();
  CpuPipelineSelection selected_source =
      CpuPipelineSelection::selected(job, 17u);
  const CpuPipelineSelection selected = std::move(selected_source);
  if (selected.disposition() != CpuPipelineSelectionDisposition::Selected ||
      !selected.status() || selected.job() != job || selected.step() != 17u) {
    return 4;
  }
  if (selected_source.disposition() !=
          CpuPipelineSelectionDisposition::Failed ||
      selected_source.status().reason() !=
          rund::compute::Reason::PipelineInvalid ||
      selected_source.job() != nullptr || selected_source.step() != 0u) {
    return 5;
  }

  const CpuPipelineSelection complete = CpuPipelineSelection::complete();
  if (complete.disposition() != CpuPipelineSelectionDisposition::Complete ||
      !complete.status() || complete.job() != nullptr ||
      complete.step() != 0u) {
    return 6;
  }
  return 0;
}

} // namespace

[[nodiscard]] int
CheckUnknownCompletionProfileIdentity(rund::compute::Device &device,
                                      const Backend backend) {
  if (const int selection = CheckCpuPipelineSelectionContract();
      selection != 0) {
    return 10 + selection;
  }
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
