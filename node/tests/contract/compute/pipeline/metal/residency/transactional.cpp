#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU) &&                                    \
    !defined(RUND_NODE_TEST_BACKEND_VULKAN)

namespace rund_node_test_pipeline {

[[nodiscard]] int CheckTransactionalStageRollback() {
  using namespace rund::compute;
  auto opened = open(Target::metal());
  if (!opened) {
    return opened.reason() == Reason::AdapterUnavailable ? 0 : 1;
  }
  Device device = std::move(opened).value();
  constexpr std::array<std::int32_t, 8u> input{1, 2, 3, 4, 5, 6, 7, 8};
  auto source = device.upload<std::int32_t>(input);
  auto state_buffer = device.buffer<std::int32_t>(input.size());
  auto program =
      on(device)
          .map<std::int32_t>("metal-residency-admission", input.size(),
                             [](auto value) { return value + 1; })
          .compile();
  if (!source || !state_buffer || !program) {
    return 2;
  }
  auto prepared = pipeline(device)
                      .state(*source, *state_buffer)
                      .then(*program, read(*source), write(*state_buffer))
                      .commit()
                      .prepare();
  if (!prepared) {
    return 3;
  }
  const std::shared_ptr<detail::PipelineState> state =
      detail::PipelineStateAccess::state(*prepared);
  if (state == nullptr || state->device == nullptr ||
      state->device->ops == nullptr ||
      state->device->ops->residency.prepare_residency_selection == nullptr ||
      !state->transactional) {
    return 4;
  }
  bool supported = false;
  if (!rund::node::accel::detail::QueryPreparedKernelPipelineResidency(
          state->prepared, supported)) {
    return 5;
  }
  if (!supported) {
    return 0;
  }
  const PipelinePlan before_plan = prepared->plan();
  const MemoryStats before_memory = prepared->memory();
  const DevicePipelineMemoryReport before_report = device.pipeline_memory();
  if (before_report.available_bytes <= 1u) {
    return 6;
  }
  rund::storage::Reservation capacity_gate =
      state->device->pipeline_memory_budget.reserve(
          before_report.available_bytes);
  std::shared_ptr<void> primary_candidate;
  std::uint64_t primary_bytes = 0u;
  const auto primary =
      rund::node::accel::detail::StagePreparedKernelPipelineResidency(
          state->prepared, primary_candidate, primary_bytes);
  rund::node::accel::detail::PreparedKernelPipeline invalid{};
  std::shared_ptr<void> alternate_candidate;
  std::uint64_t alternate_bytes = 0u;
  const auto alternate =
      rund::node::accel::detail::StagePreparedKernelPipelineResidency(
          invalid, alternate_candidate, alternate_bytes);
  primary_candidate.reset();
  alternate_candidate.reset();
  if (!capacity_gate.refund() || !primary || primary_bytes == 0u || alternate ||
      state->residency_submission_memory || prepared->plan() != before_plan ||
      !SameCurrent(prepared->memory(), before_memory) ||
      device.pipeline_memory().committed_bytes !=
          before_report.committed_bytes ||
      device.pipeline_memory().preparing_bytes != 0u) {
    return 7;
  }

  // Leave positive headroom so admission stages the native candidates before
  // rejecting their actual retained charge. Driver allocation sizes from a
  // different preparation are not an exact capacity oracle.
  capacity_gate = state->device->pipeline_memory_budget.reserve(
      before_report.available_bytes - 1u);
  if (!capacity_gate) {
    return 9;
  }
  const Status rejected =
      state->device->ops->residency.prepare_residency_selection(*state);
  if (!capacity_gate.refund() || rejected ||
      rejected.reason() != Reason::DevicePipelineMemoryCapacity ||
      state->residency_submission_memory || prepared->plan() != before_plan ||
      !SameCurrent(prepared->memory(), before_memory) ||
      device.pipeline_memory().committed_bytes !=
          before_report.committed_bytes ||
      device.pipeline_memory().preparing_bytes != 0u) {
    return 10;
  }

  const Status retried =
      state->device->ops->residency.prepare_residency_selection(*state);
  const std::uint64_t admitted =
      state->residency_submission_memory.usage().allocated_bytes;
  const PipelinePlan after_plan = prepared->plan();
  const MemoryStats after_memory = prepared->memory();
  const std::uint64_t publication =
      state->publication == nullptr
          ? 0u
          : state->publication->publication_memory.usage().allocated_bytes;
  const std::uint64_t total_admission =
      state->private_memory.usage().allocated_bytes + publication + admitted;
  if (!retried || admitted == 0u ||
      !state->residency_submission_memory.committed() ||
      !AddedSubmissionBytes(before_plan, after_plan, admitted) ||
      after_plan.physical_bytes != after_plan.peak_bytes ||
      after_plan.committed_peak_bytes != total_admission ||
      device.pipeline_memory().committed_bytes != total_admission ||
      after_memory.device.current != before_memory.device.current + admitted ||
      !prepared->run()) {
    return 8;
  }
  return 0;
}

} // namespace rund_node_test_pipeline

#endif
