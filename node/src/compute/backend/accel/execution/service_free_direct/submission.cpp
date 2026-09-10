#include "internal.hpp"

namespace rund::compute::detail::accel_backend::service_free_direct {

Status submit(Run &run) noexcept {
  namespace accel = node::accel::detail;
  if (run.pipeline == nullptr || run.registration == nullptr ||
      !run.lease.has_value() || !*run.lease || !run.preparation) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const auto registration = run.registration->snapshot();
  if (registration.proof == nullptr || registration.count == 0u) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const accel::ServiceFreeDirectRequest request{
      .proof = registration.proof,
      .lowering = run.preparation.lowering,
      .admission = std::static_pointer_cast<const void>(run.pipeline),
      .token = run.lease->token(),
      .generation = run.lease->generation(),
      .nonce = run.lease->owner(),
      .final = complete,
      .user = &run,
  };
  if (!accel::service_free_direct_request_valid(run.preparation.capability,
                                                request)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const rund::AccelCheck submitted = run.preparation.submit(request);
  if (!submitted.ok) {
    return Status::fail(
        project_reason(submitted.reason, Reason::BackendFailed));
  }
  run.done.wait(false, std::memory_order_acquire);
  if (run.callback_count != 1u ||
      !accel::service_free_direct_final_valid(request, run.final)) {
    return Status::fail(Reason::CompletionInvalid);
  }
  return run.final.check.ok
             ? Status::success()
             : Status::fail(project_reason(run.final.check.reason,
                                           Reason::BackendFailed));
}

} // namespace rund::compute::detail::accel_backend::service_free_direct
