#include "local.hpp"

#include "../../target/selection.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/accel/kernel/fault.hpp"
#include "src/compute/device/state.hpp"

#include <memory>

namespace rund_node_test_pipeline {

[[nodiscard]] int CheckNativeDeviceLoss(rund::compute::Device &device,
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
  constexpr std::array<std::int32_t, 4u> initial{1, 2, 3, 4};
  constexpr std::array<std::int32_t, 4u> once{2, 3, 4, 5};
  constexpr std::array<std::int32_t, 4u> twice{3, 4, 5, 6};
  auto advance =
      on(device)
          .map<std::int32_t>("pipeline-device-loss-advance", initial.size(),
                             [](auto value) { return value + 1; })
          .compile();
  auto published = Upload(device, initial);
  auto pending = device.buffer<std::int32_t>(initial.size());
  if (!advance || !published || !pending) {
    return 1;
  }
  auto prepared = pipeline(device)
                      .profile(PipelineProfile::Steps)
                      .state(*published, *pending)
                      .then(*advance, read(*published), write(*pending))
                      .commit()
                      .prepare();
  if (!prepared || !prepared->run() || prepared->generation() != 1u) {
    return 2;
  }
  std::array<PipelineStepProfile, 1u> successful_rows{};
  const auto successful_profile = prepared->profile(successful_rows);
  if (!successful_profile || !successful_rows[0].execution.available() ||
      !ProfileMemoryReconciles(*successful_profile, successful_rows)) {
    return 2;
  }
  std::array<std::int32_t, initial.size()> observed{};
  if (!ReadExact(*prepared, *pending, observed) || observed != once) {
    return 3;
  }
  auto saved_result = prepared->snapshot();
  if (!saved_result || saved_result->generation() != 1u ||
      saved_result->hash() == 0u) {
    return 4;
  }
  const StateSnapshot saved = *saved_result;

  const std::shared_ptr<detail::DeviceState> &device_state =
      detail::DeviceAccess::state(device);
  detail::AccelDeviceState *const native =
      device_state == nullptr ? nullptr : detail::accel_device(*device_state);
  if (native == nullptr) {
    return 5;
  }
  if (backend == Backend::Vulkan) {
    if (!rund::node::accel::detail::InjectNativeDeviceLostOnce(native->pick)) {
      return 5;
    }
    const auto lost_snapshot = prepared->snapshot();
    if (lost_snapshot || lost_snapshot.reason() != Reason::DeviceLost) {
      return 5;
    }
  }
  if (!rund::node::accel::detail::InjectNativeDeviceLostOnce(native->pick)) {
    return 5;
  }
  const Status lost = prepared->run();
  const Stats lost_stats = prepared->stats();
  std::array<PipelineStepProfile, 1u> lost_rows{};
  const auto lost_profile = prepared->profile(lost_rows);
  if (lost || lost.reason() != Reason::DeviceLost || prepared->poisoned() ||
      prepared->generation() != 1u || lost_stats.command_submits != 1u ||
      lost_stats.publication.generation != 1u ||
      lost_stats.publication.commit_count != 1u ||
      lost_stats.publication.discard_count != 1u ||
      lost_stats.publication.device_loss_count != 1u || !lost_profile ||
      lost_profile->execution.pipeline.verified_step_count != 0u ||
      lost_profile->execution.pipeline.failed_step_index !=
          PipelineStats::no_failed_step ||
      lost_rows[0].execution.available() || lost_rows[0].timing.available() ||
      lost_profile->instrumentation_command_count != 0u ||
      lost_profile->instrumentation_byte_count != 0u ||
      !ProfileMemoryReconciles(*lost_profile, lost_rows)) {
    return 6;
  }

  // The injected terminal is projected only after a real native submission.
  // The failed tick targeted the alternate physical state Buffer, so a direct
  // low-level read of the previously published Buffer proves it was not
  // replaced even though the lost Pipeline correctly closes resident I/O.
  const std::shared_ptr<detail::BufferState> &pending_state =
      detail::BufferAccess::state(*pending);
  const detail::AccelBufferState *const published_native =
      pending_state == nullptr ? nullptr : detail::accel_buffer(*pending_state);
  std::array<std::int32_t, initial.size()> retained{};
  if (published_native == nullptr ||
      !rund::node::accel::DownloadAccelBuffer(native->context,
                                              published_native->buffer,
                                              retained.data(), sizeof(retained))
           .ok ||
      retained != once) {
    return 7;
  }

  const Status rerun = prepared->run();
  const Status reread = prepared->read(*pending, observed);
  const auto resnapshot = prepared->snapshot();
  const Status in_place_restore = prepared->restore(saved);
  if (rerun || rerun.reason() != Reason::DeviceLost || reread ||
      reread.reason() != Reason::DeviceLost || resnapshot ||
      resnapshot.reason() != Reason::DeviceLost || in_place_restore ||
      in_place_restore.reason() != Reason::DeviceLost) {
    return 8;
  }

  auto replacement = open(rund::node::test_contract::target_for(backend, 2u));
  if (!replacement) {
    return 9;
  }
  auto replacement_advance =
      on(*replacement)
          .map<std::int32_t>("pipeline-device-loss-advance", initial.size(),
                             [](auto value) { return value + 1; })
          .compile();
  auto replacement_first = replacement->buffer<std::int32_t>(initial.size());
  auto replacement_second = replacement->buffer<std::int32_t>(initial.size());
  if (!replacement_advance || !replacement_first || !replacement_second) {
    return 10;
  }
  auto restored = pipeline(*replacement)
                      .state(*replacement_first, *replacement_second)
                      .then(*replacement_advance, read(*replacement_first),
                            write(*replacement_second))
                      .restore(saved)
                      .commit()
                      .prepare();
  if (!restored) {
    return 11;
  }
  const Status restored_run = restored->run();
  const bool restored_read =
      ReadExact(*restored, *replacement_second, observed);
  if (restored->generation() != 2u ||
      restored->fingerprint() != saved.fingerprint() || !restored_run ||
      !restored_read || observed != twice ||
      restored->stats().publication.restore_byte_count !=
          sizeof(initial) * 2u) {
    return 11;
  }
  return 0;
#endif
}

} // namespace rund_node_test_pipeline
