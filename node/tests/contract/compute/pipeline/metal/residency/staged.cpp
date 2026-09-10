#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU) &&                                    \
    !defined(RUND_NODE_TEST_BACKEND_VULKAN)

namespace rund_node_test_pipeline {

[[nodiscard]] int CheckStagedLoopExecution(const std::size_t epochs) {
  using namespace rund::compute;
  const std::size_t elements = epochs * 2u;
  constexpr std::size_t page_elements = 1u;
  auto opened = open(Target::metal());
  if (!opened) {
    return opened.reason() == Reason::AdapterUnavailable ? 0 : 1;
  }
  Device device = std::move(opened).value();
  auto program = on(device)
                     .map<std::int32_t>("metal-window-execution", page_elements,
                                        [](auto value) { return value + 1; })
                     .compile();
  auto input_backing =
      std::make_shared<AdmissionBacking>(elements * sizeof(std::int32_t));
  auto output_backing =
      std::make_shared<AdmissionBacking>(elements * sizeof(std::int32_t));
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
  const Status status = prepared->run();
  const Stats stats = prepared->stats();
  if (!status || stats.pipeline.residency.epoch_count != epochs ||
      stats.command_submits != 1u || stats.command_inflight_peak != 1u ||
      stats.pipeline.residency.window_handoff_count != 1u ||
      stats.pipeline.residency.window_batch_count != 1u ||
      stats.pipeline.residency.window_queue_call_count != 1u ||
      stats.transfer_submissions.host_to_device != 0u ||
      stats.transfer_submissions.device_to_host != 0u ||
      stats.transfer_submissions.device_to_device != 0u ||
      stats.uploaded_bytes != 0u || stats.downloaded_bytes != 0u ||
      input_backing->callbacks() != elements ||
      output_backing->callbacks() != elements || !output_backing->all_i32(1)) {
    std::fprintf(
        stderr,
        "metal staged-loop execution status=%u epochs=%llu submits=%llu "
        "peak=%llu "
        "window=%llu/%llu/%llu transfer=%llu/%llu/%llu bytes=%llu/%llu "
        "input=%llu output=%llu values=%u\n",
        static_cast<unsigned>(status.reason()),
        static_cast<unsigned long long>(stats.pipeline.residency.epoch_count),
        static_cast<unsigned long long>(stats.command_submits),
        static_cast<unsigned long long>(stats.command_inflight_peak),
        static_cast<unsigned long long>(
            stats.pipeline.residency.window_handoff_count),
        static_cast<unsigned long long>(
            stats.pipeline.residency.window_batch_count),
        static_cast<unsigned long long>(
            stats.pipeline.residency.window_queue_call_count),
        static_cast<unsigned long long>(
            stats.transfer_submissions.host_to_device),
        static_cast<unsigned long long>(
            stats.transfer_submissions.device_to_host),
        static_cast<unsigned long long>(
            stats.transfer_submissions.device_to_device),
        static_cast<unsigned long long>(stats.uploaded_bytes),
        static_cast<unsigned long long>(stats.downloaded_bytes),
        static_cast<unsigned long long>(input_backing->callbacks()),
        static_cast<unsigned long long>(output_backing->callbacks()),
        static_cast<unsigned>(output_backing->all_i32(1)));
    return 3;
  }
  return 0;
}

[[nodiscard]] int CheckVirtualAdmissionRollback() {
  using namespace rund::compute;
  constexpr std::size_t elements = 67u;
  constexpr std::size_t page_elements = 16u;
  constexpr std::size_t bytes = elements * sizeof(std::int32_t);
  auto opened = open(Target::metal());
  if (!opened) {
    return opened.reason() == Reason::AdapterUnavailable ? 0 : 1;
  }
  Device device = std::move(opened).value();
  auto program =
      on(device)
          .map<std::int32_t>("metal-residency-budget", page_elements,
                             [](auto value) { return value + 1; })
          .compile();
  auto input_backing = std::make_shared<AdmissionBacking>(bytes);
  auto output_backing = std::make_shared<AdmissionBacking>(bytes);
  auto input = virtual_buffer<std::int32_t>(elements, input_backing);
  auto output = virtual_buffer<std::int32_t>(elements, output_backing);
  const auto state = detail::DeviceAccess::state(device);
  const auto before_report = device.pipeline_memory();
  if (!program || !input || !output || state == nullptr ||
      before_report.available_bytes <= 1u) {
    return 2;
  }
  auto gate = state->pipeline_memory_budget.reserve(
      before_report.available_bytes - 1u);
  if (!gate) {
    return 3;
  }
  const MemoryStats before_memory = device.memory();
  auto rejected =
      virtual_pipeline(*program, *input, *output, ResidencyConfig{});
  const MemoryStats after_memory = device.memory();
  if (!gate.refund() || rejected ||
      rejected.reason() != Reason::PipelineMemoryBudget ||
      !SameCurrent(before_memory, after_memory) ||
      device.pipeline_memory().committed_bytes != before_report.committed_bytes ||
      device.pipeline_memory().preparing_bytes != 0u ||
      input_backing->callbacks() != 0u || output_backing->callbacks() != 0u) {
    return 4;
  }
  auto retried =
      virtual_pipeline(*program, *input, *output, ResidencyConfig{});
  if (!retried || !retried->run() || !output_backing->all_i32(1)) {
    return 5;
  }
  return 0;
}

} // namespace rund_node_test_pipeline

#endif
