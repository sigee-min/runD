#include "internal.hpp"

#include <cstdio>

namespace rund_node_test_pipeline_residency::device_vsm_test::binary_detail {

[[nodiscard]] bool
RunBinary(const rund::compute::Backend backend,
          const std::shared_ptr<rund::compute::detail::DeviceState> &device,
          const std::uint64_t pages, const ActualPrepare prepare,
          const ActualQueueCount queue_count, std::uint64_t &retained) {
  auto prepared = PrepareBinary(backend, device, pages, prepare, queue_count);
  if (prepared == nullptr) {
    return false;
  }
  auto &test_case = *prepared;
  std::uint64_t queue_before = 0u;
  std::uint64_t queue_after = 0u;
  const bool lease_valid = static_cast<bool>(test_case.wait->lease);
  const bool request_valid = accel::device_vsm_request_valid(
      test_case.preparation.capability, test_case.wait->request);
  const bool queue_before_valid =
      queue_count(test_case.native->pick, queue_before);
  const rund::AccelCheck submitted =
      lease_valid && request_valid && queue_before_valid
          ? test_case.preparation.submit(test_case.wait->request)
          : rund::AccelCheck{false, "not_submitted"};
  const bool callback_ready =
      submitted.ok && wait_detail::wait(*test_case.wait);
  const bool queue_after_valid =
      submitted.ok && callback_ready &&
      queue_count(test_case.native->pick, queue_after);
  if (submitted.ok && !callback_ready) {
    wait_detail::quarantine(*test_case.wait);
    const std::uint64_t callbacks = wait_detail::count(*test_case.wait);
    std::fprintf(
        stderr,
        "DeviceVsm binary backend=%u q=%llu queue=%llu/%llu "
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
    std::lock_guard lock{test_case.wait->gate};
    final_valid = test_case.wait->final_ready && !test_case.wait->abandoned &&
                  accel::device_vsm_final_valid(test_case.wait->request,
                                                test_case.wait->final);
  }
  const bool close_ready =
      final_valid && queue_after_valid && queue_after == queue_before + 1u;
  wait_detail::Result close_result = wait_detail::Result::Failed;
  if ((callback_ready && test_case.wait->final.terminal ==
                             accel::DeviceVsmTerminal::UnknownMayWrite) ||
      close_ready) {
    close_result = CloseAuthority(*test_case.wait, test_case.wait->final);
  }
  const bool closed = close_result == wait_detail::Result::Closed;
  const bool quarantined = close_result == wait_detail::Result::Quarantined;
  if (!lease_valid || !request_valid || !queue_before_valid || !submitted.ok ||
      !queue_after_valid || !final_valid || queue_after != queue_before + 1u ||
      !closed || (callback_ready && !test_case.wait->final.check.ok)) {
    if (!submitted.ok) {
      static_cast<void>(wait_detail::cleanup_before_submit(*test_case.wait));
    } else if (!closed && !quarantined &&
               test_case.wait->final.terminal !=
                   accel::DeviceVsmTerminal::UnknownMayWrite) {
      wait_detail::quarantine(*test_case.wait);
    }
    const wait_detail::Snapshot snapshot =
        wait_detail::snapshot(*test_case.wait);
    std::fprintf(
        stderr,
        "DeviceVsm binary backend=%u q=%llu proof=%u prepared=%u/%s "
        "lease=%u request=%u queue_valid=%u/%u submitted=%u/%s "
        "callback=%llu final=%u/%u closed=%u reason=%s terminal=%u "
        "native=%u/%u/%llu mapped=%u acquired=%u may_write=%u "
        "epochs=%llu/%llu failed_page=%llu "
        "counts=%llu/%llu/%llu/%llu/%llu/%llu "
        "counters0_7=%llu/%llu/%llu/%llu/%llu/%llu/x/%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned long long>(pages),
        static_cast<unsigned>(test_case.proof != nullptr),
        static_cast<unsigned>(static_cast<bool>(test_case.preparation)),
        test_case.preparation.capability.check.reason == nullptr
            ? "null"
            : test_case.preparation.capability.check.reason,
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
  return ValidateBinaryOutputAndEvidence(test_case, retained);
}

} // namespace rund_node_test_pipeline_residency::device_vsm_test::binary_detail
