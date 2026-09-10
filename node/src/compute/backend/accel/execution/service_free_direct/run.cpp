#include "internal.hpp"

#include "../../../../pipeline/claim.hpp"
#include "../../../../pipeline/local.hpp"
#include "../../local.hpp"

#include <new>

namespace rund::compute::detail::accel_backend {

Status
run_service_free_direct_product(const std::shared_ptr<PipelineState> &pipeline,
                                bool &selected) noexcept {
  selected = false;
  if (!valid_pipeline(pipeline) || pipeline->transactional ||
      pipeline->device == nullptr || pipeline->device->residency == nullptr) {
    return Status::fail(Reason::BackendUnsupported);
  }
  const node::accel::detail::ServiceFreeDirectPreparation preparation =
      node::accel::detail::PreparePreparedKernelPipelineServiceFreeDirect(
          pipeline->prepared);
  if (!preparation) {
    return Status::fail(Reason::BackendUnsupported);
  }
  selected = true;
  try {
    auto run = std::make_shared<service_free_direct::Run>();
    run->pipeline = pipeline;
    run->preparation = preparation;
    std::unique_lock state_lock{pipeline->gate, std::try_to_lock};
    if (!state_lock.owns_lock()) {
      return Status::fail(Reason::PipelineBusy);
    }
    pipeline->service_free_direct = ServiceFreeDirectProductEvidence{
        .public_handoff_count = 1u, .selected = true};
    const Status started = start_pipeline(*pipeline);
    if (!started) {
      return started;
    }
    run->registration = residency::register_direct_recurrence_proof(
        pipeline->device->residency,
        preparation.lowering == nullptr
            ? nullptr
            : node::accel::detail::PreparedKernelPipelineServiceFreeDirectProof(
                  pipeline->prepared));
    if (run->registration == nullptr) {
      publish_pipeline_terminal(
          *pipeline, PipelineTerminal{.reason = Reason::PipelineInvalid,
                                      .publication_suppressed = true});
      return Status::fail(Reason::PipelineInvalid);
    }
    run->lease.emplace(
        pipeline->device->residency->authority().direct_recurrences().begin_direct_recurrence(
            run->registration->request()));
    if (!run->lease.has_value() || !*run->lease) {
      static_cast<void>(run->registration->release());
      publish_pipeline_terminal(
          *pipeline, PipelineTerminal{.reason = Reason::PipelineBusy,
                                      .publication_suppressed = true});
      return Status::fail(Reason::PipelineBusy);
    }
    const Status submitted = service_free_direct::submit(*run);
    if (!run->done.load(std::memory_order_acquire)) {
      service_free_direct::cancel(*run, submitted);
      return submitted;
    }
    return service_free_direct::close(*run);
  } catch (const std::bad_alloc &) {
    return Status::fail(Reason::PipelineCapacity);
  }
}

} // namespace rund::compute::detail::accel_backend
