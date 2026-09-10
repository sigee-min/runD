#include "interface/api.hpp"

#include "model.hpp"

#include <algorithm>
#include <new>
#include <utility>

namespace rund::node::accel::detail {
namespace {

struct ServiceFreeDirectSubmission final {
  ServiceFreeDirectRequest request{};
};

[[nodiscard]] ServiceFreeDirectCapability
capability_for(const prepared::PipelineState &pipeline,
               const ServiceFreeDirectProof &proof) noexcept {
  if (pipeline.ops == nullptr || pipeline.backend == nullptr ||
      pipeline.ops->service_free_direct_capability == nullptr ||
      pipeline.service_free_direct == nullptr ||
      pipeline.service_free_direct.get() != &proof ||
      pipeline.state_count != 1u || pipeline.states == nullptr ||
      pipeline.states[0] == nullptr) {
    return {};
  }
  ServiceFreeDirectCapability capability =
      pipeline.ops->service_free_direct_capability(pipeline.backend, proof);
  return capability;
}

void CompleteServiceFreeDirect(void *const raw,
                               PreparedPipelineEvidence &&observed) noexcept {
  auto *const submission = static_cast<ServiceFreeDirectSubmission *>(raw);
  if (submission == nullptr || submission->request.proof == nullptr ||
      submission->request.final == nullptr ||
      submission->request.user == nullptr) {
    return;
  }
  ServiceFreeDirectRequest &request = submission->request;
  const bool authenticated =
      observed.submitted && observed.control_observed && observed.control_valid;
  // This callback follows an accepted native submission. Without one exact
  // acquired control receipt, no layer can prove that the fused command made
  // no writes, even if a backend adapter labelled its transport terminal
  // Known. Preserve uncertainty instead of fabricating a reusable row.
  const bool unknown = observed.terminal == NativeTerminal::UnknownMayWrite ||
                       !authenticated;
  const std::uint64_t completed =
      observed.check.ok ? request.proof->iterations
      : unknown || !observed.control_valid
          ? 0u
          : std::min<std::uint64_t>(request.proof->iterations,
                                    observed.control.verified_prefix);
  ServiceFreeDirectFinal final{
      .check = authenticated
                   ? observed.check
                   : rund::AccelCheck{false, "compute_backend_failed"},
      .terminal = unknown ? ServiceFreeDirectTerminal::UnknownMayWrite
                          : ServiceFreeDirectTerminal::Known,
      .evidence =
          ServiceFreeDirectEvidence{
              .proof = request.proof->identity,
              .token = request.token,
              .generation = request.generation,
              .nonce = request.nonce,
              .iterations = request.proof->iterations,
              .completed_iterations = completed,
              .native_submit_count = 1u,
              .epoch_native_submit_count = 0u,
              .payload_dispatch_count = 1u,
              .host_service_turn_count = 0u,
              .host_epoch_callback_count = 0u,
              .final_callback_count = 1u,
              .completed_ns = observed.shared.run.time.accel_kernel_ns,
              .may_write = observed.submitted || observed.check.ok || unknown,
          },
      .profile = observed.profile,
      .active_step_count = observed.active_step_count,
      .profile_owner = observed.profile.observed
                           ? std::static_pointer_cast<const void>(
                                 request.lowering)
                           : std::shared_ptr<const void>{},
  };
  if (!service_free_direct_final_valid(request, final)) {
    final = ServiceFreeDirectFinal{
        .check = {false, "compute_backend_failed"},
        .terminal = ServiceFreeDirectTerminal::UnknownMayWrite,
        .evidence =
            ServiceFreeDirectEvidence{
                .proof = request.proof->identity,
                .token = request.token,
                .generation = request.generation,
                .nonce = request.nonce,
                .iterations = request.proof->iterations,
                .completed_iterations = 0u,
                .native_submit_count = 1u,
                .epoch_native_submit_count = 0u,
                .payload_dispatch_count = 1u,
                .host_service_turn_count = 0u,
                .host_epoch_callback_count = 0u,
                .final_callback_count = 1u,
                .may_write = true,
            },
    };
  }
  const ServiceFreeDirectFinalCompletion completion =
      std::exchange(request.final, nullptr);
  void *const user = std::exchange(request.user, nullptr);
  completion(user, std::move(final));
}

[[nodiscard]] rund::AccelCheck SubmitPreparedServiceFreeDirect(
    const ServiceFreeDirectRequest &request) noexcept {
  auto *const pipeline =
      static_cast<prepared::PipelineState *>(request.lowering.get());
  if (pipeline == nullptr || request.proof == nullptr ||
      pipeline->service_free_direct != request.proof ||
      !service_free_direct_request_valid(
          capability_for(*pipeline, *request.proof), request)) {
    return {false, "accel_kernel_pipeline_invalid"};
  }
  try {
    auto submission = std::make_shared<ServiceFreeDirectSubmission>();
    submission->request = request;
    const PreparedKernelPipeline prepared{
        .owner = request.lowering,
        .ok = true,
    };
    return SubmitPreparedKernelPipeline(
        pipeline->context, prepared, submission, CompleteServiceFreeDirect,
        submission.get(), KernelTiming::Submission,
        PipelineSubmitMode::Standard);
  } catch (const std::bad_alloc &) {
    return {false, "compute_pipeline_capacity"};
  }
}

} // namespace

std::shared_ptr<const ServiceFreeDirectProof>
PreparedKernelPipelineServiceFreeDirectProof(
    const PreparedKernelPipeline &prepared) noexcept {
  const auto *const pipeline =
      static_cast<const prepared::PipelineState *>(prepared.owner.get());
  return prepared.ok && pipeline != nullptr ? pipeline->service_free_direct
                                            : nullptr;
}

ServiceFreeDirectPreparation PreparePreparedKernelPipelineServiceFreeDirect(
    const PreparedKernelPipeline &prepared) noexcept {
  const auto *const pipeline =
      static_cast<const prepared::PipelineState *>(prepared.owner.get());
  if (!prepared.ok || pipeline == nullptr ||
      pipeline->service_free_direct == nullptr) {
    return {};
  }
  ServiceFreeDirectCapability capability =
      capability_for(*pipeline, *pipeline->service_free_direct);
  if (!service_free_direct_capable(capability)) {
    return ServiceFreeDirectPreparation{.capability = capability};
  }
  return ServiceFreeDirectPreparation{
      .capability = capability,
      .lowering = prepared.owner,
      .submit = SubmitPreparedServiceFreeDirect,
  };
}

} // namespace rund::node::accel::detail
