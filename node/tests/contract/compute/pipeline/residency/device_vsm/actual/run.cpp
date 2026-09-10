#include "internal.hpp"

#include "src/accel/context/transfer.hpp"
#include "src/compute/device/residency/execution/device_vsm/registration.hpp"
#include "src/compute/device/state.hpp"

#include <node/accel/context.hpp>

#include <array>
#include <cstdio>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace rund_node_test_pipeline_residency::device_vsm_test {

namespace residency = rund::compute::detail::residency;

[[nodiscard]] bool
RunActual(const rund::compute::Backend backend,
          const std::shared_ptr<rund::compute::detail::DeviceState> &device,
          const std::uint64_t pages, const ActualPrepare prepare,
          const ActualQueueCount queue_count, std::uint64_t &retained) {
  using namespace rund::node::accel::detail;
  const auto *const native = device == nullptr
                                 ? nullptr
                                 : rund::compute::detail::accel_device(*device);
  constexpr std::uint64_t PayloadElements = 16u;
  const std::size_t logical_elements =
      static_cast<std::size_t>(pages * PayloadElements - 3u);
  std::vector<std::uint32_t> input_values(logical_elements);
  std::vector<std::uint32_t> zero(logical_elements, 0u);
  const auto reject = [backend, pages](const char *const phase) noexcept {
    std::fprintf(stderr, "DeviceVsm actual backend=%u q=%llu phase=%s\n",
                 static_cast<unsigned>(backend),
                 static_cast<unsigned long long>(pages), phase);
    return false;
  };
  for (std::size_t index = 0u; index < input_values.size(); ++index) {
    input_values[index] = static_cast<std::uint32_t>(index * 17u + pages);
  }
  if (native == nullptr || prepare == nullptr || queue_count == nullptr) {
    return reject("owner");
  }
  const rund::AccelBuffer input = rund::node::accel::CreateAccelBuffer(
      native->context,
      rund::AccelBufferDesc{.scalar_width_bytes = sizeof(std::uint32_t),
                            .count = logical_elements,
                            .usage = rund::BufferUsage::ReadOnly});
  const rund::AccelBuffer output = rund::node::accel::CreateAccelBuffer(
      native->context,
      rund::AccelBufferDesc{.scalar_width_bytes = sizeof(std::uint32_t),
                            .count = logical_elements,
                            .usage = rund::BufferUsage::WriteOnly});
  std::array<UploadEntry, 2u> uploads{
      UploadEntry{.buffer = &input,
                  .data = input_values.data(),
                  .bytes = input_values.size() * sizeof(std::uint32_t)},
      UploadEntry{.buffer = &output,
                  .data = zero.data(),
                  .bytes = zero.size() * sizeof(std::uint32_t)},
  };
  std::array<UploadRoute, 2u> routes{};
  if (!input || !output ||
      !UploadAccelBuffers(native->context, uploads, routes,
                          TransferCompletion::Complete)
           .check.ok) {
    return reject("staging");
  }
  routes[0u] = ProjectAccelBufferRoute(native->context, input);
  routes[1u] = ProjectAccelBufferRoute(native->context, output);
  if (routes[0u].handle == nullptr || routes[1u].handle == nullptr) {
    return reject("projection");
  }
  std::shared_ptr<accel::DeviceVsmProof> proof =
      BuildProof(ApiFor(backend), pages, routes[0u], routes[1u]);
  if (proof == nullptr || !accel::device_vsm_proof_valid(*proof)) {
    if (proof != nullptr) {
      std::fprintf(
          stderr,
          "DeviceVsm proof backend=%u artifact=%u/%u plan=%u/%u "
          "dispatch=%llu/%llu params=%llu/%llu input=%llu/%llu/%u/%llu/%llu "
          "output=%llu/%llu/%u/%llu/%llu geometry=%llu/%llu/%llu/%llu/%u "
          "width=%u\n",
          static_cast<unsigned>(backend),
          static_cast<unsigned>(proof->artifact != nullptr),
          static_cast<unsigned>(proof->artifact != nullptr &&
                                proof->artifact->ok),
          static_cast<unsigned>(proof->plan.ok),
          static_cast<unsigned>(proof->artifact != nullptr &&
                                proof->artifact->key.api == proof->plan.api),
          static_cast<unsigned long long>(proof->plan.dispatch_count),
          static_cast<unsigned long long>(proof->window_count),
          static_cast<unsigned long long>(proof->parameter_bytes),
          static_cast<unsigned long long>(proof->plan.param_bytes),
          static_cast<unsigned long long>(proof->residents.rows[0u].backing.id),
          static_cast<unsigned long long>(
              proof->residents.rows[0u].backing.bytes),
          proof->residents.rows[0u].backing.usage,
          static_cast<unsigned long long>(
              proof->residents.rows[0u].backing.element_bytes),
          static_cast<unsigned long long>(
              proof->residents.rows[0u].backing.count),
          static_cast<unsigned long long>(proof->residents.rows[1u].backing.id),
          static_cast<unsigned long long>(
              proof->residents.rows[1u].backing.bytes),
          proof->residents.rows[1u].backing.usage,
          static_cast<unsigned long long>(
              proof->residents.rows[1u].backing.element_bytes),
          static_cast<unsigned long long>(
              proof->residents.rows[1u].backing.count),
          static_cast<unsigned long long>(proof->geometry.logical_bytes),
          static_cast<unsigned long long>(proof->geometry.payload_bytes),
          static_cast<unsigned long long>(proof->geometry.frame_bytes),
          static_cast<unsigned long long>(proof->geometry.page_count),
          proof->geometry.element_bytes, proof->width);
    }
    return reject("proof");
  }
  const accel::DeviceVsmPreparation preparation =
      accel::PrepareDeviceVsm(native->pick, proof);
  if (!preparation || preparation.capability.width != proof->width) {
    return reject(preparation.capability.check.reason);
  }
  retained = preparation.capability.retained_bytes;
  const std::shared_ptr<residency::DeviceVsmRegistration> registration =
      residency::register_device_vsm_proof(device->residency, proof);
  residency::DirectRecurrenceLease lease =
      registration == nullptr
          ? residency::DirectRecurrenceLease{}
          : device->residency->authority().direct_recurrences().begin_direct_recurrence(
                registration->request());
  auto wait = std::make_shared<wait_detail::Owner>(std::move(lease));
  wait->device = device;
  wait->registry = device->residency;
  wait->registration = registration;
  wait->self = wait;
  wait->request = accel::DeviceVsmRequest{
      .proof = proof,
      .lowering = preparation.lowering,
      .admission = registration,
      .token = wait->lease.token(),
      .generation = wait->lease.generation(),
      .nonce = wait->lease.owner(),
      .submission_control = &wait->submission_control,
      .final = wait_detail::complete,
      .user = wait.get(),
  };
  std::uint64_t queue_before = 0u;
  std::uint64_t queue_after = 0u;
  const bool lease_valid = static_cast<bool>(wait->lease);
  const bool request_valid =
      accel::device_vsm_request_valid(preparation.capability, wait->request);
  const bool queue_before_valid = queue_count(native->pick, queue_before);
  const rund::AccelCheck submitted =
      lease_valid && request_valid && queue_before_valid
          ? preparation.submit(wait->request)
          : rund::AccelCheck{false, "not_submitted"};
  const bool callback_ready = submitted.ok && wait_detail::wait(*wait);
  const bool queue_after_valid =
      submitted.ok && callback_ready && queue_count(native->pick, queue_after);
  if (submitted.ok && !callback_ready) {
    wait_detail::quarantine(*wait);
    const std::uint64_t callbacks = wait_detail::count(*wait);
    std::fprintf(
        stderr,
        "DeviceVsm actual backend=%u q=%llu queue=%llu/%llu "
        "lease=%u request=%u queue_valid=%u/%u submitted=%u/%s "
        "callback=%llu timeout=1 final=unavailable terminal=unavailable "
        "native=unavailable mapped=unavailable acquired=unavailable "
        "may_write=unavailable epochs=unavailable failed_page=unavailable "
        "counts=unavailable counters0_7=unavailable\n",
        static_cast<unsigned>(backend), static_cast<unsigned long long>(pages),
        static_cast<unsigned long long>(queue_before),
        static_cast<unsigned long long>(queue_after),
        static_cast<unsigned>(lease_valid),
        static_cast<unsigned>(request_valid),
        static_cast<unsigned>(queue_before_valid),
        static_cast<unsigned>(queue_after_valid),
        static_cast<unsigned>(submitted.ok), submitted.reason,
        static_cast<unsigned long long>(callbacks));
    return false;
  }
  bool final_valid = false;
  if (callback_ready) {
    std::lock_guard lock{wait->gate};
    final_valid = wait->final_ready && !wait->abandoned &&
                  accel::device_vsm_final_valid(wait->request, wait->final);
  }
  const bool close_ready =
      final_valid && queue_after_valid && queue_after == queue_before + 1u;
  wait_detail::Result close_result = wait_detail::Result::Failed;
  if ((callback_ready &&
       wait->final.terminal == accel::DeviceVsmTerminal::UnknownMayWrite) ||
      close_ready) {
    close_result = CloseAuthority(*wait, wait->final);
  }
  const bool closed = close_result == wait_detail::Result::Closed;
  const bool quarantined = close_result == wait_detail::Result::Quarantined;
  if (!lease_valid || !request_valid || !queue_before_valid || !submitted.ok ||
      !queue_after_valid || !final_valid || queue_after != queue_before + 1u ||
      !closed || (callback_ready && !wait->final.check.ok)) {
    if (!submitted.ok) {
      static_cast<void>(wait_detail::cleanup_before_submit(*wait));
    } else if (!closed && !quarantined &&
               wait->final.terminal !=
                   accel::DeviceVsmTerminal::UnknownMayWrite) {
      wait_detail::quarantine(*wait);
    }
    const wait_detail::Snapshot snapshot = wait_detail::snapshot(*wait);
    std::fprintf(
        stderr,
        "DeviceVsm actual backend=%u q=%llu queue=%llu/%llu "
        "lease=%u request=%u queue_valid=%u/%u submitted=%u/%s "
        "callback=%llu final=%u/%u closed=%u reason=%s terminal=%u "
        "native=%u/%u/%llu mapped=%u acquired=%u may_write=%u "
        "epochs=%llu/%llu failed_page=%llu "
        "counts=%llu/%llu/%llu/%llu/%llu/%llu "
        "counters0_7=%llu/%llu/%llu/%llu/%llu/%llu/x/%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned long long>(pages),
        static_cast<unsigned long long>(queue_before),
        static_cast<unsigned long long>(queue_after),
        static_cast<unsigned>(lease_valid),
        static_cast<unsigned>(request_valid),
        static_cast<unsigned>(queue_before_valid),
        static_cast<unsigned>(queue_after_valid),
        static_cast<unsigned>(submitted.ok), submitted.reason,
        static_cast<unsigned long long>(snapshot.callback_count),
        static_cast<unsigned>(snapshot.final.check.ok),
        static_cast<unsigned>(final_valid), static_cast<unsigned>(closed),
        snapshot.final.check.reason == nullptr ? "null"
                                               : snapshot.final.check.reason,
        static_cast<unsigned>(snapshot.final.terminal),
        static_cast<unsigned>(snapshot.final.evidence.native_check_ok),
        snapshot.final.evidence.native_check_code,
        static_cast<unsigned long long>(
            snapshot.final.evidence.native_check_reason),
        static_cast<unsigned>(snapshot.final.evidence.result_mapped),
        static_cast<unsigned>(snapshot.final.evidence.result_acquired),
        static_cast<unsigned>(snapshot.final.evidence.may_write),
        static_cast<unsigned long long>(
            snapshot.final.evidence.generated_epochs),
        static_cast<unsigned long long>(
            snapshot.final.evidence.completed_epochs),
        static_cast<unsigned long long>(snapshot.final.evidence.failed_page),
        static_cast<unsigned long long>(
            snapshot.final.evidence.native_submit_count),
        static_cast<unsigned long long>(
            snapshot.final.evidence.epoch_native_submit_count),
        static_cast<unsigned long long>(
            snapshot.final.evidence.payload_dispatch_count),
        static_cast<unsigned long long>(
            snapshot.final.evidence.host_service_turn_count),
        static_cast<unsigned long long>(
            snapshot.final.evidence.host_epoch_callback_count),
        static_cast<unsigned long long>(
            snapshot.final.evidence.final_callback_count),
        static_cast<unsigned long long>(
            snapshot.final.evidence.generated_epochs),
        static_cast<unsigned long long>(
            snapshot.final.evidence.forecasted_pages),
        static_cast<unsigned long long>(snapshot.final.evidence.promoted_pages),
        static_cast<unsigned long long>(
            snapshot.final.evidence.completed_epochs),
        static_cast<unsigned long long>(snapshot.final.evidence.drained_pages),
        static_cast<unsigned long long>(
            snapshot.final.evidence.persisted_pages),
        static_cast<unsigned long long>(snapshot.final.evidence.failed_page));
    return false;
  }
  wait_detail::release_self(*wait);
  std::vector<std::uint32_t> observed(logical_elements);
  const AccelTransfer downloaded =
      DownloadAccelBufferMeasured(native->context, output, observed.data(),
                                  observed.size() * sizeof(std::uint32_t));
  if (!downloaded.check.ok) {
    return reject(downloaded.check.reason);
  }
  for (std::size_t index = 0u; index < observed.size(); ++index) {
    if (observed[index] != input_values[index] + 1u) {
      return reject("output");
    }
  }
  const accel::DeviceVsmEvidence &evidence = wait->final.evidence;
  std::uint32_t expected_ring_reuse = 0u;
  std::uint32_t expected_ring_checksum = 0u;
  std::uint32_t expected_ring_round_trips = 0u;
  std::uint64_t expected_ring_state_bytes = 0u;
  std::uint64_t expected_ring_scratch_bytes = 0u;
  const bool ring_exact =
      accel::device_vsm_ring_schedule_expected(proof->geometry, proof->width,
                                               expected_ring_reuse,
                                               expected_ring_checksum) &&
      accel::device_vsm_ring_storage_expected(
          proof->geometry, proof->width, proof->residents.count,
          expected_ring_state_bytes, expected_ring_scratch_bytes) &&
      accel::device_vsm_ring_round_trips_expected(
          proof->geometry, proof->residents.count, expected_ring_round_trips) &&
      evidence.ring_reuse_transitions == expected_ring_reuse &&
      evidence.ring_schedule_checksum == expected_ring_checksum &&
      evidence.ring_round_trips == expected_ring_round_trips &&
      evidence.ring_state_bytes == expected_ring_state_bytes &&
      evidence.ring_scratch_bytes == expected_ring_scratch_bytes;
  const bool valid =
      ring_exact && evidence.page_count == pages &&
      evidence.generated_epochs == pages &&
      evidence.completed_epochs == pages &&
      evidence.forecasted_pages == pages && evidence.promoted_pages == pages &&
      evidence.drained_pages == pages && evidence.persisted_pages == pages &&
      evidence.native_submit_count == 1u &&
      evidence.epoch_native_submit_count == 0u &&
      evidence.payload_dispatch_count == 1u &&
      evidence.host_service_turn_count == 0u &&
      evidence.host_epoch_callback_count == 0u &&
      evidence.final_callback_count == 1u;
  std::fprintf(
      stderr,
      "DeviceVsm actual backend=%u q=%llu valid=%u submit=%llu "
      "epoch_submit=%llu host_callback=%llu final=%llu ring=%llu/%llu/%llu "
      "retained=%llu\n",
      static_cast<unsigned>(backend), static_cast<unsigned long long>(pages),
      static_cast<unsigned>(valid),
      static_cast<unsigned long long>(evidence.native_submit_count),
      static_cast<unsigned long long>(evidence.epoch_native_submit_count),
      static_cast<unsigned long long>(evidence.host_epoch_callback_count),
      static_cast<unsigned long long>(evidence.final_callback_count),
      static_cast<unsigned long long>(evidence.ring_round_trips),
      static_cast<unsigned long long>(evidence.ring_state_bytes),
      static_cast<unsigned long long>(evidence.ring_scratch_bytes),
      static_cast<unsigned long long>(retained));
  return valid;
}

} // namespace rund_node_test_pipeline_residency::device_vsm_test
