#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU) &&                                    \
    !defined(RUND_NODE_TEST_BACKEND_VULKAN)

namespace rund_node_test_pipeline {

[[nodiscard]] int CheckSelectedNativeTerminalAndDeviceLoss() {
  using namespace rund::compute;
  constexpr std::size_t elements = 16u;
  constexpr std::size_t bytes = elements * sizeof(std::int32_t);
  auto opened = open(Target::metal());
  if (!opened) {
    return opened.reason() == Reason::AdapterUnavailable ? 0 : 1;
  }
  Device device = std::move(opened).value();
  auto program =
      on(device)
          .map<std::int32_t>("metal-residency-native-terminal", elements,
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
  const std::uint64_t output_callbacks = output_backing->callbacks();
  if (native == nullptr ||
      !rund::node::accel::detail::InjectNativeDeviceLostOnce(native->pick)) {
    return 3;
  }
  const Status lost = prepared->run();
  const Stats stats = prepared->stats();
  const Status poisoned = prepared->run();
  if (lost || lost.reason() != Reason::DeviceLost ||
      poisoned.reason() != Reason::DeviceLost || stats.command_submits != 1u ||
      stats.publication.device_loss_count != 1u ||
      stats.pipeline.residency.page_in_count != 0u ||
      output_backing->callbacks() != output_callbacks) {
    std::fprintf(
        stderr,
        "metal residency native terminal lost=%u poison=%u submits=%llu "
        "losses=%llu page_in=%llu input_callbacks=%llu "
        "output_callbacks=%llu/%llu\n",
        static_cast<unsigned>(lost.reason()),
        static_cast<unsigned>(poisoned.reason()),
        static_cast<unsigned long long>(stats.command_submits),
        static_cast<unsigned long long>(stats.publication.device_loss_count),
        static_cast<unsigned long long>(stats.pipeline.residency.page_in_count),
        static_cast<unsigned long long>(input_backing->callbacks()),
        static_cast<unsigned long long>(output_backing->callbacks()),
        static_cast<unsigned long long>(output_callbacks));
    return 4;
  }
  return 0;
}

[[nodiscard]] int CheckQueueTerminalLossQuarantine() {
  using namespace rund::compute;
  constexpr std::size_t elements = 16u;
  constexpr std::size_t bytes = elements * sizeof(std::int32_t);
  auto opened = open(Target::metal());
  if (!opened) {
    return opened.reason() == Reason::AdapterUnavailable ? 0 : 1;
  }
  Device device = std::move(opened).value();
  auto program =
      on(device)
          .map<std::int32_t>("metal-residency-terminal-loss", elements,
                             [](auto value) { return value + 1; })
          .compile();
  auto input_backing = std::make_shared<AdmissionBacking>(bytes);
  auto output_backing = std::make_shared<AdmissionBacking>(bytes);
  auto peer_input_backing = std::make_shared<AdmissionBacking>(bytes);
  auto peer_output_backing = std::make_shared<AdmissionBacking>(bytes);
  auto input = virtual_buffer<std::int32_t>(elements, input_backing);
  auto output = virtual_buffer<std::int32_t>(elements, output_backing);
  auto peer_input = virtual_buffer<std::int32_t>(elements, peer_input_backing);
  auto peer_output =
      virtual_buffer<std::int32_t>(elements, peer_output_backing);
  auto prepared =
      program && input && output
          ? virtual_pipeline(*program, *input, *output, ResidencyConfig{})
          : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                Reason::PipelineInvalid);
  auto peer = program && peer_input && peer_output
                  ? virtual_pipeline(*program, *peer_input, *peer_output,
                                     ResidencyConfig{})
                  : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                        Reason::PipelineInvalid);
  if (!prepared || !peer || !prepared->run() || !peer->run()) {
    return 2;
  }
  const std::shared_ptr<detail::DeviceState> &state =
      detail::DeviceAccess::state(device);
  detail::AccelDeviceState *const native =
      state == nullptr ? nullptr : detail::accel_device(*state);
  const std::uint64_t output_callbacks = output_backing->callbacks();
  const std::uint64_t peer_input_callbacks = peer_input_backing->callbacks();
  const std::uint64_t peer_output_callbacks = peer_output_backing->callbacks();
  const std::uint64_t committed = device.pipeline_memory().committed_bytes;
  if (native == nullptr || committed == 0u ||
      !rund::node::accel::detail::InjectNativeResidencyTerminalLossOnce(
          native->pick)) {
    return 3;
  }
  const auto started = std::chrono::steady_clock::now();
  const Status lost = prepared->run();
  const auto elapsed = std::chrono::steady_clock::now() - started;
  const Stats lost_stats = prepared->stats();
  const Status same_owner = prepared->run();
  const Status peer_rejected = peer->run();

  auto new_input_backing = std::make_shared<AdmissionBacking>(bytes);
  auto new_output_backing = std::make_shared<AdmissionBacking>(bytes);
  auto new_input = virtual_buffer<std::int32_t>(elements, new_input_backing);
  auto new_output = virtual_buffer<std::int32_t>(elements, new_output_backing);
  auto newly_prepared =
      new_input && new_output
          ? virtual_pipeline(*program, *new_input, *new_output,
                             ResidencyConfig{})
          : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                Reason::PipelineInvalid);

  if (lost.reason() != Reason::DeviceLost ||
      elapsed >= std::chrono::seconds{2} ||
      same_owner.reason() != Reason::DeviceLost ||
      peer_rejected.reason() != Reason::DeviceLost || newly_prepared ||
      newly_prepared.reason() != Reason::DeviceLost ||
      lost_stats.command_submits != 1u ||
      lost_stats.command_inflight_peak != 1u ||
      lost_stats.publication.device_loss_count != 1u ||
      output_backing->callbacks() != output_callbacks ||
      peer_input_backing->callbacks() != peer_input_callbacks ||
      peer_output_backing->callbacks() != peer_output_callbacks ||
      new_input_backing->callbacks() != 0u ||
      new_output_backing->callbacks() != 0u ||
      device.pipeline_memory().committed_bytes != committed) {
    std::fprintf(
        stderr,
        "metal residency terminal loss lost=%u same=%u peer=%u new=%u "
        "elapsed_ms=%lld submits=%llu peak=%llu losses=%llu "
        "callbacks=%llu/%llu "
        "peer_callbacks=%llu/%llu new_callbacks=%llu/%llu "
        "committed=%llu/%llu\n",
        static_cast<unsigned>(lost.reason()),
        static_cast<unsigned>(same_owner.reason()),
        static_cast<unsigned>(peer_rejected.reason()),
        static_cast<unsigned>(newly_prepared.reason()),
        static_cast<long long>(
            std::chrono::duration_cast<std::chrono::milliseconds>(elapsed)
                .count()),
        static_cast<unsigned long long>(lost_stats.command_submits),
        static_cast<unsigned long long>(lost_stats.command_inflight_peak),
        static_cast<unsigned long long>(
            lost_stats.publication.device_loss_count),
        static_cast<unsigned long long>(input_backing->callbacks()),
        static_cast<unsigned long long>(output_backing->callbacks()),
        static_cast<unsigned long long>(peer_input_backing->callbacks()),
        static_cast<unsigned long long>(peer_output_backing->callbacks()),
        static_cast<unsigned long long>(new_input_backing->callbacks()),
        static_cast<unsigned long long>(new_output_backing->callbacks()),
        static_cast<unsigned long long>(
            device.pipeline_memory().committed_bytes),
        static_cast<unsigned long long>(committed));
    return 4;
  }
  // Leaving this scope drops every public owner. The quarantined native owner
  // intentionally remains self-retained, so destruction performs no wait and
  // cannot release an allocator or frame still reachable by the lost command.
  return 0;
}

} // namespace rund_node_test_pipeline

#endif
