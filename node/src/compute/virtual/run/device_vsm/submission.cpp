#include "internal.hpp"

#include <cstring>
#include <utility>

namespace rund::compute::detail::device_vsm_product_detail {

void test_phase(DeviceVsmProductRun &run,
                const DeviceVsmTestPhase value) noexcept {
  if (run.state == nullptr || run.state->device_vsm_test_barrier == nullptr) {
    return;
  }
  run.state->device_vsm_test_barrier->publish(value);
}

void complete(void *const raw,
              node::accel::detail::DeviceVsmFinal &&final) noexcept {
  auto *const run = static_cast<DeviceVsmProductRun *>(raw);
  if (run == nullptr) {
    return;
  }
  test_phase(*run, DeviceVsmTestPhase::Callback);
  if (run->state != nullptr && run->state->device_vsm_test_barrier != nullptr) {
    run->state->device_vsm_test_barrier->wait_released();
  }
  std::uint64_t expected = 0u;
  if (!run->callback_count.compare_exchange_strong(
          expected, 1u, std::memory_order_acq_rel,
          std::memory_order_acquire)) {
    run->callback_count.store(2u, std::memory_order_release);
    return;
  }
  const Status final_status = status_from(final.check);
  if (run->owner != nullptr && run->owner->evidence != nullptr) {
    run->owner->evidence->final_terminal = final.terminal;
    run->owner->evidence->final_reason = final_status.reason();
    run->owner->evidence->final_code = final_status.code();
    run->owner->evidence->native = final.evidence;
    run->owner->evidence->final_received = true;
  }
  run->final = std::move(final);
  const bool lost =
      run->final.terminal ==
          node::accel::detail::DeviceVsmTerminal::UnknownMayWrite ||
      (run->final.check.reason != nullptr &&
       std::strcmp(run->final.check.reason, "compute_device_lost") == 0);
  if (lost) {
    quarantine_owner(run->owner);
    run->poison_pipeline = true;
  }
  run->done.store(true, std::memory_order_release);
  run->done.notify_one();
  if (run->wake != nullptr) {
    run->wake(run->wake_user);
  }
}

Status submit(DeviceVsmProductRun &run) noexcept {
  namespace accel = node::accel::detail;
  const auto fail = [&run](const Status status) noexcept {
    test_phase(run, DeviceVsmTestPhase::Failed);
    return status;
  };
  if (run.owner == nullptr || run.registration == nullptr ||
      !run.lease.has_value() || !*run.lease || !run.owner->preparation) {
    return fail(Status::fail(Reason::PipelineInvalid));
  }
  const accel::DeviceVsmRequest request{
      .proof = run.owner->proof,
      .lowering = run.owner->preparation.lowering,
      .admission = run.registration,
      .token = run.lease->token(),
      .generation = run.lease->generation(),
      .nonce = run.lease->owner(),
      .submission_control = &run.submission_control,
      .final = complete,
      .user = &run,
  };
  run.request = request;
  if (!accel::device_vsm_request_valid(run.owner->preparation.capability,
                                       request)) {
    return fail(Status::fail(Reason::PipelineInvalid));
  }
  const rund::AccelCheck submitted = run.owner->preparation.submit(request);
  if (!submitted.ok) {
    return fail(status_from(submitted));
  }
  if (run.submission_control.count() != 1u) {
    return fail(Status::fail(Reason::CompletionInvalid));
  }
  run.owner->submitted = true;
  test_phase(run, DeviceVsmTestPhase::Accepted);
  if (!run.wait_for_callback) {
    return Status::success();
  }
  run.done.wait(false, std::memory_order_acquire);
  if (run.callback_count.load(std::memory_order_acquire) != 1u ||
      run.submission_control.count() != 1u ||
      !accel::device_vsm_final_valid(request, run.final)) {
    run.final = unknown_final(run);
    return Status::fail(Reason::DeviceLost);
  }
  return status_from(run.final.check);
}

} // namespace rund::compute::detail::device_vsm_product_detail
