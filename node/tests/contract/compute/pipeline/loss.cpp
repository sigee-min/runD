#include "local.hpp"

#include "../../target/selection.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/accel/context/transfer.hpp"
#include "src/accel/kernel/fault.hpp"
#include "src/compute/device/state.hpp"

#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>

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
    // Generic one-slice upload retains the queued command-slot contract, while
    // checkpoint restore explicitly requires completion before publication.
    // The test fault is consumed only by the latter synchronous boundary.
    auto policy_buffer = Upload(device, initial);
    const std::shared_ptr<detail::BufferState> policy_state =
        policy_buffer ? detail::BufferAccess::state(*policy_buffer) : nullptr;
    detail::AccelBufferState *const policy_native =
        policy_state == nullptr ? nullptr : detail::accel_buffer(*policy_state);
    const rund::node::accel::detail::UploadEntry policy_request{
        .buffer = policy_native == nullptr ? nullptr : &policy_native->buffer,
        .data = initial.data(),
        .bytes = sizeof(initial),
    };
    rund::node::accel::detail::UploadRoute policy_route{};
    if (policy_native == nullptr ||
        !rund::node::accel::detail::InjectNativeDeviceLostOnce(native->pick)) {
      return 24;
    }
    const auto queued = rund::node::accel::detail::UploadAccelBuffers(
        native->context, std::span{&policy_request, 1u},
        std::span{&policy_route, 1u},
        rund::node::accel::detail::TransferCompletion::Queued);
    const auto completed =
        queued.check.ok
            ? rund::node::accel::detail::UploadAccelBuffers(
                  native->context, std::span{&policy_request, 1u},
                  std::span{&policy_route, 1u},
                  rund::node::accel::detail::TransferCompletion::Complete)
            : rund::node::accel::detail::AccelTransfer{};
    if (!queued.check.ok || completed.check.ok ||
        std::string_view{completed.check.reason} != "compute_device_lost") {
      return 25;
    }

    auto export_first = Upload(device, initial);
    auto export_second = device.buffer<std::int32_t>(initial.size());
    auto export_pipeline =
        export_first && export_second
            ? pipeline(device)
                  .state(*export_first, *export_second)
                  .then(*advance, read(*export_first), write(*export_second))
                  .commit()
                  .prepare()
            : Result<Pipeline>::fail(Reason::PipelineInvalid);
    auto export_storage_result =
        export_pipeline
            ? export_pipeline->snapshot_storage()
            : Result<SnapshotStorage>::fail(Reason::PipelineInvalid);
    auto export_latest =
        export_pipeline
            ? export_pipeline->latest_device_state()
            : Result<LatestDeviceState>::fail(Reason::PipelineInvalid);
    if (!export_pipeline || !export_storage_result || !export_latest) {
      return 15;
    }
    SnapshotStorage export_storage = std::move(export_storage_result).value();
    if (!export_pipeline->run() ||
        !export_pipeline->snapshot_into(export_storage)) {
      return 15;
    }
    const std::uint64_t retained_generation = export_storage.generation();
    const std::uint64_t retained_hash = export_storage.hash();
    const CheckpointStats retained_stats = export_pipeline->checkpoint_stats();
    if (!rund::node::accel::detail::InjectNativeDeviceLostOnce(native->pick)) {
      return 16;
    }
    const Status export_lost = export_pipeline->snapshot_into(export_storage);
    const CheckpointStats lost_export_stats =
        export_pipeline->checkpoint_stats();
    const auto export_resnapshot = export_pipeline->snapshot();
    const Status export_restore = export_pipeline->restore(export_storage);
    const Status export_rerun = export_pipeline->run();
    if (export_lost || export_lost.reason() != Reason::DeviceLost ||
        export_storage.generation() != retained_generation ||
        export_storage.hash() != retained_hash || export_latest->valid() ||
        export_latest->generation() != retained_generation ||
        lost_export_stats.reusable_snapshot_count !=
            retained_stats.reusable_snapshot_count ||
        lost_export_stats.reusable_snapshot_byte_count !=
            retained_stats.reusable_snapshot_byte_count ||
        lost_export_stats.reusable_snapshot_hash !=
            retained_stats.reusable_snapshot_hash ||
        export_resnapshot || export_resnapshot.reason() != Reason::DeviceLost ||
        export_restore || export_restore.reason() != Reason::DeviceLost ||
        export_rerun || export_rerun.reason() != Reason::DeviceLost ||
        export_pipeline->poisoned()) {
      return 16;
    }

    auto restore_first = device.buffer<std::int32_t>(initial.size());
    auto restore_second = device.buffer<std::int32_t>(initial.size());
    auto restore_pipeline =
        restore_first && restore_second
            ? pipeline(device)
                  .state(*restore_first, *restore_second)
                  .then(*advance, read(*restore_first), write(*restore_second))
                  .commit()
                  .prepare()
            : Result<Pipeline>::fail(Reason::PipelineInvalid);
    auto restore_latest =
        restore_pipeline
            ? restore_pipeline->latest_device_state()
            : Result<LatestDeviceState>::fail(Reason::PipelineInvalid);
    if (!restore_pipeline || !restore_latest ||
        !rund::node::accel::detail::InjectNativeDeviceLostOnce(native->pick)) {
      return 17;
    }
    const Status restore_lost = restore_pipeline->restore(saved);
    const Status restore_read =
        restore_pipeline->read(*restore_first, observed);
    const auto restore_snapshot = restore_pipeline->snapshot();
    const Status restore_again = restore_pipeline->restore(saved);
    const Status restore_run = restore_pipeline->run();
    if (restore_lost || restore_lost.reason() != Reason::DeviceLost) {
      return 18;
    }
    if (restore_latest->valid() || restore_latest->generation() != 0u) {
      return 19;
    }
    if (restore_read || restore_read.reason() != Reason::DeviceLost) {
      return 20;
    }
    if (restore_snapshot || restore_snapshot.reason() != Reason::DeviceLost) {
      return 21;
    }
    if (restore_again || restore_again.reason() != Reason::DeviceLost) {
      return 22;
    }
    if (restore_run || restore_run.reason() != Reason::DeviceLost) {
      return 23;
    }
  }

  auto sealed_input = Upload(device, initial);
  auto sealed_output = device.buffer<std::int32_t>(initial.size());
  auto sealed =
      sealed_input && sealed_output
          ? pipeline(device)
                .sealed_repetitions<8u>()
                .then(*advance, read(*sealed_input), write(*sealed_output))
                .prepare()
          : Result<Pipeline>::fail(Reason::PipelineInvalid);
  if (!sealed) {
    return 12;
  }
  if (!rund::node::accel::detail::InjectNativeDeviceLostOnce(native->pick)) {
    return 13;
  }
  const Status sealed_lost = sealed->run();
  const Stats sealed_lost_stats = sealed->stats();
  if (sealed_lost || sealed_lost.reason() != Reason::DeviceLost ||
      !sealed->poisoned() || sealed->generation() != 0u ||
      sealed_lost_stats.command_submits != 1u ||
      sealed_lost_stats.publication.generation != 0u ||
      sealed_lost_stats.publication.device_loss_count != 1u ||
      sealed_lost_stats.pipeline.sealed_repetition_count != 8u ||
      sealed_lost_stats.pipeline.coalesced_repetition_count != 0u) {
    return 14;
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
