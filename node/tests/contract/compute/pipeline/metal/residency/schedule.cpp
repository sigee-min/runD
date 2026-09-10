#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU) &&                                    \
    !defined(RUND_NODE_TEST_BACKEND_VULKAN)

namespace rund_node_test_pipeline {

[[nodiscard]] int CheckScheduleTerminalLossQuarantine() {
  using namespace rund::compute;
  constexpr std::size_t epochs = 5u;
  constexpr std::size_t elements = epochs * 2u;
  constexpr std::size_t bytes = elements * sizeof(std::int32_t);
  auto opened = open(Target::metal());
  if (!opened) {
    return opened.reason() == Reason::AdapterUnavailable ? 0 : 1;
  }
  Device device = std::move(opened).value();
  auto program = on(device)
                     .map<std::int32_t>("metal-schedule-terminal-loss", 1u,
                                        [](auto value) { return value + 1; })
                     .compile();
  auto input_backing = std::make_shared<AdmissionBacking>(bytes);
  auto output_backing = std::make_shared<AdmissionBacking>(bytes);
  auto input = virtual_buffer<std::int32_t>(elements, input_backing);
  auto output = virtual_buffer<std::int32_t>(elements, output_backing);
  auto prepared =
      program && input && output
          ? virtual_pipeline(*program, *input, *output, ResidencyConfig{})
          : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                Reason::PipelineInvalid);
  if (!prepared || !prepared->run()) {
    return 2;
  }
  const std::shared_ptr<detail::DeviceState> &state =
      detail::DeviceAccess::state(device);
  detail::AccelDeviceState *const native =
      state == nullptr ? nullptr : detail::accel_device(*state);
  if (native == nullptr ||
      !rund::node::accel::detail::InjectNativeResidencyTerminalLossOnce(
          native->pick)) {
    return 3;
  }
  ScheduleRouteScope schedule_route{*state};
  if (!schedule_route) {
    return 3;
  }
  const auto started = std::chrono::steady_clock::now();
  const Status lost = prepared->run();
  const auto elapsed = std::chrono::steady_clock::now() - started;
  const Stats stats = prepared->stats();
  const Status retry = prepared->run();
  if (lost.reason() != Reason::DeviceLost ||
      retry.reason() != Reason::DeviceLost ||
      elapsed >= std::chrono::seconds{2} || stats.command_submits != epochs ||
      stats.command_inflight_peak < 1u ||
      stats.pipeline.residency.window_handoff_count != 1u ||
      stats.pipeline.residency.window_batch_count != epochs ||
      stats.pipeline.residency.window_queue_call_count != epochs ||
      stats.publication.device_loss_count != 1u) {
    std::fprintf(
        stderr,
        "metal schedule loss lost=%u retry=%u elapsed_ms=%lld "
        "submits=%llu peak=%llu window=%llu/%llu/%llu loss=%llu\n",
        static_cast<unsigned>(lost.reason()),
        static_cast<unsigned>(retry.reason()),
        static_cast<long long>(
            std::chrono::duration_cast<std::chrono::milliseconds>(elapsed)
                .count()),
        static_cast<unsigned long long>(stats.command_submits),
        static_cast<unsigned long long>(stats.command_inflight_peak),
        static_cast<unsigned long long>(
            stats.pipeline.residency.window_handoff_count),
        static_cast<unsigned long long>(
            stats.pipeline.residency.window_batch_count),
        static_cast<unsigned long long>(
            stats.pipeline.residency.window_queue_call_count),
        static_cast<unsigned long long>(stats.publication.device_loss_count));
    return 4;
  }
  return 0;
}

[[nodiscard]] int CheckScheduleBudgetFallback() {
  using namespace rund::compute;
  constexpr std::size_t epochs = 5u;
  constexpr std::size_t elements = epochs * 2u;
  constexpr std::size_t bytes = elements * sizeof(std::int32_t);
  auto opened = open(Target::metal());
  if (!opened) {
    return opened.reason() == Reason::AdapterUnavailable ? 0 : 1;
  }
  Device device = std::move(opened).value();
  auto program = on(device)
                     .map<std::int32_t>("metal-schedule-budget", 1u,
                                        [](auto value) { return value + 1; })
                     .compile();
  auto input_backing = std::make_shared<AdmissionBacking>(bytes);
  auto output_backing = std::make_shared<AdmissionBacking>(bytes);
  auto input = detail::make_virtual_buffer(
      elements, sizeof(std::int32_t), detail::Type::I32, {}, input_backing);
  auto output = detail::make_virtual_buffer(
      elements, sizeof(std::int32_t), detail::Type::I32, {}, output_backing);
  auto prepared =
      program && input && output
          ? detail::prepare_virtual_pipeline(
                detail::ProgramAccess::state(*program),
                std::move(input).value(), std::move(output).value(),
                ResidencyConfig{})
          : Result<std::shared_ptr<detail::VirtualPipelineState>>::fail(
                Reason::PipelineInvalid);
  if (!prepared || prepared.value()->pipeline == nullptr ||
      prepared.value()->alternate_pipeline == nullptr ||
      prepared.value()->pipeline->device == nullptr ||
      prepared.value()->pipeline->residency_pool == nullptr) {
    return 2;
  }
  const std::shared_ptr<detail::VirtualPipelineState> virtual_state =
      prepared.value();
  const std::array<std::shared_ptr<detail::PipelineState>, 2u> pipelines{
      virtual_state->pipeline, virtual_state->alternate_pipeline};
  const execution::SealResult sealed =
      ExecutionPlan(*virtual_state->pipeline->residency_pool, elements);
  if (!sealed || sealed.plan.epoch_count() != epochs) {
    return 3;
  }

  const DevicePipelineMemoryReport baseline = device.pipeline_memory();
  detail::PipelineExecutionSchedulePrepared probe{};
  const Status probed = detail::prepare_pipeline_execution_schedule(
      sealed.plan, pipelines, probe);
  const std::uint64_t retained = probe.lowering.retained_bytes;
  if (!probed || !probe || retained == 0u ||
      baseline.available_bytes < retained ||
      device.pipeline_memory().committed_bytes !=
          baseline.committed_bytes + retained) {
    return 4;
  }
  probe = {};
  if (device.pipeline_memory().committed_bytes != baseline.committed_bytes) {
    return 5;
  }

  const std::uint64_t blocked_bytes =
      baseline.available_bytes - (retained - 1u);
  rund::storage::Reservation blocker =
      virtual_state->pipeline->device->pipeline_memory_budget.reserve(
          blocked_bytes);
  detail::PipelineExecutionSchedulePrepared rejected{};
  const Status rejection = detail::prepare_pipeline_execution_schedule(
      sealed.plan, pipelines, rejected);
  const DevicePipelineMemoryReport rejected_report = device.pipeline_memory();
  ScheduleRouteScope schedule_route{*virtual_state->pipeline->device};
  if (!schedule_route) {
    return 6;
  }
  const Status fallback = detail::run_virtual_pipeline(virtual_state);
  const Stats fallback_stats = detail::virtual_pipeline_stats(virtual_state);
  if (!blocker || rejection.reason() != Reason::DevicePipelineMemoryCapacity ||
      rejected || !fallback ||
      rejected_report.committed_bytes != baseline.committed_bytes ||
      rejected_report.preparing_bytes != blocked_bytes ||
      fallback_stats.pipeline.residency.window_handoff_count != 1u ||
      fallback_stats.pipeline.residency.window_batch_count != epochs ||
      fallback_stats.pipeline.residency.window_queue_call_count != epochs ||
      input_backing->callbacks() != elements ||
      output_backing->callbacks() != elements || !output_backing->all_i32(1) ||
      device.pipeline_memory().committed_bytes != baseline.committed_bytes ||
      !blocker.refund()) {
    std::fprintf(
        stderr,
        "metal schedule budget blocker=%u rejection=%u rejected=%u "
        "fallback=%u report=%llu/%llu baseline=%llu/%llu "
        "window=%llu/%llu/%llu callbacks=%llu/%llu values=%u live=%llu\n",
        static_cast<unsigned>(static_cast<bool>(blocker)),
        static_cast<unsigned>(rejection.reason()),
        static_cast<unsigned>(static_cast<bool>(rejected)),
        static_cast<unsigned>(fallback.reason()),
        static_cast<unsigned long long>(rejected_report.committed_bytes),
        static_cast<unsigned long long>(rejected_report.preparing_bytes),
        static_cast<unsigned long long>(baseline.committed_bytes),
        static_cast<unsigned long long>(blocked_bytes),
        static_cast<unsigned long long>(
            fallback_stats.pipeline.residency.window_handoff_count),
        static_cast<unsigned long long>(
            fallback_stats.pipeline.residency.window_batch_count),
        static_cast<unsigned long long>(
            fallback_stats.pipeline.residency.window_queue_call_count),
        static_cast<unsigned long long>(input_backing->callbacks()),
        static_cast<unsigned long long>(output_backing->callbacks()),
        static_cast<unsigned>(output_backing->all_i32(1)),
        static_cast<unsigned long long>(
            device.pipeline_memory().committed_bytes));
    return 6;
  }

  detail::PipelineExecutionSchedulePrepared admitted{};
  const Status admission = detail::prepare_pipeline_execution_schedule(
      sealed.plan, pipelines, admitted);
  if (!admission || !admitted || admitted.lowering.retained_bytes != retained ||
      device.pipeline_memory().committed_bytes !=
          baseline.committed_bytes + retained) {
    return 7;
  }
  admitted = {};
  if (device.pipeline_memory().committed_bytes != baseline.committed_bytes ||
      device.pipeline_memory().preparing_bytes != 0u) {
    return 8;
  }
  return 0;
}

[[nodiscard]] int CheckScheduleKnownFailureRetry() {
  using namespace rund::compute;
  constexpr std::size_t epochs = 5u;
  constexpr std::size_t elements = epochs * 2u;
  constexpr std::size_t bytes = elements * sizeof(std::int32_t);
  auto opened = open(Target::metal());
  if (!opened) {
    return opened.reason() == Reason::AdapterUnavailable ? 0 : 1;
  }
  Device device = std::move(opened).value();
  auto program = on(device)
                     .map<std::int32_t>("metal-schedule-known-failure", 1u,
                                        [](auto value) { return value + 1; })
                     .compile();
  auto input_backing = std::make_shared<AdmissionBacking>(bytes);
  auto output_backing = std::make_shared<AdmissionBacking>(bytes);
  auto input = virtual_buffer<std::int32_t>(elements, input_backing);
  auto output = virtual_buffer<std::int32_t>(elements, output_backing);
  auto prepared =
      program && input && output
          ? virtual_pipeline(*program, *input, *output, ResidencyConfig{})
          : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                Reason::PipelineInvalid);
  if (!prepared) {
    return 2;
  }
  const std::shared_ptr<detail::DeviceState> &state =
      detail::DeviceAccess::state(device);
  if (state == nullptr) {
    return 2;
  }
  ScheduleRouteScope schedule_route{*state};
  if (!schedule_route) {
    return 2;
  }
  const std::uint64_t committed = device.pipeline_memory().committed_bytes;
  input_backing->fail_next_read();
  const Status failed = prepared->run();
  const Stats failed_stats = prepared->stats();
  const Status retry = prepared->run();
  const Stats retry_stats = prepared->stats();
  if (failed.reason() != Reason::BackendFailed || !retry ||
      failed_stats.command_submits != epochs ||
      failed_stats.pipeline.residency.window_handoff_count != 1u ||
      failed_stats.pipeline.residency.window_batch_count != epochs ||
      failed_stats.pipeline.residency.window_queue_call_count != epochs ||
      failed_stats.publication.device_loss_count != 0u ||
      retry_stats.command_submits != epochs ||
      retry_stats.pipeline.residency.window_handoff_count != 1u ||
      retry_stats.pipeline.residency.window_batch_count != epochs ||
      retry_stats.pipeline.residency.window_queue_call_count != epochs ||
      input_backing->callbacks() != elements ||
      output_backing->callbacks() != elements || !output_backing->all_i32(1) ||
      device.pipeline_memory().committed_bytes != committed) {
    std::fprintf(
        stderr,
        "metal schedule known failure failed=%u retry=%u "
        "failed_submits=%llu failed_window=%llu/%llu/%llu loss=%llu "
        "retry_submits=%llu retry_window=%llu/%llu/%llu callbacks=%llu/%llu "
        "values=%u committed=%llu/%llu\n",
        static_cast<unsigned>(failed.reason()),
        static_cast<unsigned>(retry.reason()),
        static_cast<unsigned long long>(failed_stats.command_submits),
        static_cast<unsigned long long>(
            failed_stats.pipeline.residency.window_handoff_count),
        static_cast<unsigned long long>(
            failed_stats.pipeline.residency.window_batch_count),
        static_cast<unsigned long long>(
            failed_stats.pipeline.residency.window_queue_call_count),
        static_cast<unsigned long long>(
            failed_stats.publication.device_loss_count),
        static_cast<unsigned long long>(retry_stats.command_submits),
        static_cast<unsigned long long>(
            retry_stats.pipeline.residency.window_handoff_count),
        static_cast<unsigned long long>(
            retry_stats.pipeline.residency.window_batch_count),
        static_cast<unsigned long long>(
            retry_stats.pipeline.residency.window_queue_call_count),
        static_cast<unsigned long long>(input_backing->callbacks()),
        static_cast<unsigned long long>(output_backing->callbacks()),
        static_cast<unsigned>(output_backing->all_i32(1)),
        static_cast<unsigned long long>(
            device.pipeline_memory().committed_bytes),
        static_cast<unsigned long long>(committed));
    return 3;
  }
  return 0;
}

} // namespace rund_node_test_pipeline

#endif
