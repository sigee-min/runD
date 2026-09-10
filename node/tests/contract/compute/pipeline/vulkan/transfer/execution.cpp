#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU) &&                                    \
    !defined(RUND_NODE_TEST_BACKEND_METAL) &&                                  \
    defined(RUND_NODE_HAVE_VULKAN_SDK)

#include <cstdio>

namespace rund_node_test_pipeline::vulkan_transfer {

int ExactVulkanExecutionAdapter() {
  using namespace rund::compute;
  constexpr std::array<std::uint32_t, 1u> input{7u};
  auto opened = open(Target::vulkan());
  if (!opened) {
    return opened.reason() == Reason::AdapterUnavailable ? 0 : 1;
  }
  Device device = std::move(opened).value();
  auto program =
      on(device)
          .map<std::uint32_t>("vulkan-execution-adapter", input.size(),
                              [](auto value) { return value + 1u; })
          .compile();
  auto input_backing = std::make_shared<VulkanTransferBacking>(sizeof(input));
  auto output_backing = std::make_shared<VulkanTransferBacking>(sizeof(input));
  auto input_state =
      detail::make_virtual_buffer(input.size(), sizeof(std::uint32_t),
                                  detail::Type::U32, {}, input_backing);
  auto output_state =
      detail::make_virtual_buffer(input.size(), sizeof(std::uint32_t),
                                  detail::Type::U32, {}, output_backing);
  if (!program || !input_state || !output_state) {
    return 2;
  }
  auto prepared = detail::prepare_virtual_pipeline(
      detail::ProgramAccess::state(*program), std::move(input_state).value(),
      std::move(output_state).value(), ResidencyConfig{});
  if (!prepared) {
    return 3;
  }
  const std::shared_ptr<detail::PipelineState> state =
      prepared.value()->pipeline;
  if (state == nullptr || state->device == nullptr ||
      state->device->ops == nullptr ||
      state->device->ops->residency.prepare_residency_selection == nullptr ||
      state->device->ops->residency.prepare_residency_execution == nullptr ||
      state->residency_pool == nullptr) {
    return 4;
  }
  const execution::SealResult sealed =
      VulkanExecutionPlan(*state->residency_pool, 1u);
  if (!sealed || sealed.plan.epoch_count() != 1u) {
    return 5;
  }
  const std::array<const rund::node::accel::detail::PreparedKernelPipeline *,
                   1u>
      pipelines{&state->prepared};
  execution::Owner owner{};
  const Status bound =
      state->device->ops->residency.prepare_residency_execution(
          *state->device, pipelines, owner);
  std::uint64_t retained = 1u;
  if (!bound || !owner || !owner.retained_bytes(retained) || retained != 0u) {
    return 6;
  }
  const PipelinePlan retained_plan =
      detail::virtual_pipeline_plan(prepared.value());
  const MemoryStats retained_memory =
      detail::virtual_pipeline_memory(prepared.value());
  const detail::AccelDeviceState *const accelerator =
      detail::accel_device(*state->device);
  if (accelerator == nullptr) {
    return 6;
  }
  const rund::RuntimeStats cold_native =
      rund::node::accel::ReadRuntimeStats(accelerator->pick);

  execution::Control control{};
  execution::NativeEvidence produced{};
  VulkanExecutionWait wait{};
  detail::PipelineExecutionSubmission submission{};
  Reason submit_failure = Reason::Ok;
  control.evidence = &produced;
  control.completion = CompleteVulkanExecution;
  control.user = &wait;
  const auto submit = [&](const std::uint64_t sample) noexcept {
    wait.done.store(false, std::memory_order_relaxed);
    wait.evidence = {};
    produced = {};
    control.token = sample;
    control.generation = sample;
    const Status submitted = detail::submit_pipeline_execution(
        state, owner, sealed.plan, control, submission);
    if (!submitted) {
      submit_failure = submitted.reason();
      return false;
    }
    wait.done.wait(false, std::memory_order_acquire);
    const execution::NativeEvidence &evidence = wait.evidence;
    const bool valid =
        evidence.status &&
        evidence.terminal == execution::TerminalKind::Known &&
        evidence.plan_identity == sealed.plan.identity() &&
        evidence.token == sample && evidence.generation == sample &&
        evidence.epoch_count == 1u && evidence.native_submissions == 1u &&
        evidence.native_dispatches == 1u && evidence.native_completions == 1u &&
        evidence.native_inflight_peak == 1u && evidence.failure_count == 0u &&
        !control.active.load(std::memory_order_acquire);
    submission.reset();
    return valid;
  };

  const std::uint64_t before = VulkanQueueSubmits(state);
  constexpr std::uint64_t WarmRuns = 60u;
  for (std::uint64_t sample = 1u; sample <= WarmRuns + 1u; ++sample) {
    if (!submit(sample)) {
      std::fprintf(
          stderr,
          "vulkan execution sample=%llu status=%u terminal=%u plan=%llu/%llu "
          "token=%llu generation=%llu submit=%llu dispatch=%llu "
          "complete=%llu peak=%llu failures=%zu active=%u\n",
          static_cast<unsigned long long>(sample),
          static_cast<unsigned>(wait.evidence.status.reason()),
          static_cast<unsigned>(wait.evidence.terminal),
          static_cast<unsigned long long>(wait.evidence.plan_identity),
          static_cast<unsigned long long>(sealed.plan.identity()),
          static_cast<unsigned long long>(wait.evidence.token),
          static_cast<unsigned long long>(wait.evidence.generation),
          static_cast<unsigned long long>(wait.evidence.native_submissions),
          static_cast<unsigned long long>(wait.evidence.native_dispatches),
          static_cast<unsigned long long>(wait.evidence.native_completions),
          static_cast<unsigned long long>(wait.evidence.native_inflight_peak),
          wait.evidence.failure_count,
          static_cast<unsigned>(
              control.active.load(std::memory_order_acquire)));
      std::fprintf(
          stderr,
          "vulkan execution reject=%u phase=%u transactional=%u "
          "publications=%zu windows=%zu stage=%u graph_stage=%u ports=%zu "
          "bank=%u steps=%zu k=%u\n",
          static_cast<unsigned>(submit_failure),
          static_cast<unsigned>(state->phase),
          static_cast<unsigned>(state->transactional),
          state->publications.size(), state->windows.size(),
          static_cast<unsigned>(state->residency_stage),
          state->residency_graph_stage, state->residency_port_count,
          state->residency_bank, state->steps.size(),
          state->residency_pool->layout.frame_capacity);
      execution::Node dispatch{};
      const bool projected = sealed.plan.project(
          execution::NodeId{.epoch = 0u, .phase = execution::Phase::Dispatch},
          dispatch);
      const residency::FrameRegion expected_input =
          state->residency_pool->input_regions[state->residency_bank];
      std::fprintf(
          stderr,
          "vulkan execution private=%u projected=%u in=%u/%u/%zu:%u/%u/%u "
          "out=%u/%u/%zu owner=%u publication=%u residency=%u\n",
          static_cast<unsigned>(
              detail::has_private_residency_authority(*state)),
          static_cast<unsigned>(projected), dispatch.route.source.first,
          dispatch.route.source.count, dispatch.input_count,
          expected_input.first, expected_input.count, state->residency_input,
          dispatch.route.target.first, dispatch.route.target.count,
          dispatch.output_count,
          static_cast<unsigned>(owner.native == state->prepared.owner),
          static_cast<unsigned>(state->publication != nullptr),
          static_cast<unsigned>(state->residency != nullptr));
      return 7;
    }
  }
  const std::uint64_t after = VulkanQueueSubmits(state);
  const rund::RuntimeStats warm_native =
      rund::node::accel::ReadRuntimeStats(accelerator->pick);
  const auto &cold_allocations = cold_native.run.allocations;
  const auto &warm_allocations = warm_native.run.allocations;
  if (after != before + WarmRuns + 1u ||
      detail::virtual_pipeline_plan(prepared.value()) != retained_plan ||
      detail::virtual_pipeline_memory(prepared.value()) != retained_memory ||
      cold_allocations.pipeline_compile_count !=
          warm_allocations.pipeline_compile_count ||
      cold_allocations.descriptor_pool_create_count !=
          warm_allocations.descriptor_pool_create_count ||
      cold_allocations.descriptor_set_allocate_count !=
          warm_allocations.descriptor_set_allocate_count ||
      cold_allocations.buffer_allocation_count !=
          warm_allocations.buffer_allocation_count) {
    std::fprintf(stderr,
                 "vulkan execution queue submits before=%llu after=%llu "
                 "expected=%llu\n",
                 static_cast<unsigned long long>(before),
                 static_cast<unsigned long long>(after),
                 static_cast<unsigned long long>(before + WarmRuns + 1u));
    return 8;
  }

  const execution::SealResult recurrent =
      VulkanExecutionPlan(*state->residency_pool,
                          state->residency_pool->layout.frame_capacity + 1u);
  const std::uint64_t before_reject = VulkanQueueSubmits(state);
  wait.done.store(false, std::memory_order_relaxed);
  control.token = WarmRuns + 2u;
  control.generation = WarmRuns + 2u;
  const Status recurrent_status =
      recurrent ? detail::submit_pipeline_execution(
                      state, owner, recurrent.plan, control, submission)
                : Status::fail(Reason::PipelineInvalid);
  if (!recurrent || recurrent.plan.epoch_count() != 2u ||
      recurrent_status.reason() != Reason::BackendUnsupported ||
      control.active.load(std::memory_order_acquire) || submission.active() ||
      wait.done.load(std::memory_order_acquire) ||
      VulkanQueueSubmits(state) != before_reject) {
    std::fprintf(
        stderr,
        "vulkan execution recurrent sealed=%u epochs=%llu reason=%u "
        "control=%u submission=%u callback=%u submits=%llu/%llu\n",
        static_cast<unsigned>(static_cast<bool>(recurrent)),
        static_cast<unsigned long long>(recurrent.plan.epoch_count()),
        static_cast<unsigned>(recurrent_status.reason()),
        static_cast<unsigned>(control.active.load(std::memory_order_acquire)),
        static_cast<unsigned>(submission.active()),
        static_cast<unsigned>(wait.done.load(std::memory_order_acquire)),
        static_cast<unsigned long long>(before_reject),
        static_cast<unsigned long long>(VulkanQueueSubmits(state)));
    return 9;
  }
  return 0;
}

} // namespace rund_node_test_pipeline::vulkan_transfer

#endif
