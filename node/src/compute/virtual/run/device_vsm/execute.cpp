#include "internal.hpp"


namespace rund::compute::detail {

Status start_accel_virtual_execution_device_vsm(
    VirtualPipelineState &state, const std::span<VirtualBacking *const> inputs,
    VirtualBacking &output, const VirtualRunProjection &projection,
    const VirtualExecutionDeviceVsmPrepared &prepared, Stats &stats,
    device_vsm_product_detail::DeviceVsmProductRun &run) noexcept {
  using namespace device_vsm_product_detail;
  const std::shared_ptr<DeviceVsmProductOwner> owner =
      execution_owner(state, prepared);
  if (owner == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (inputs.size() != owner->input_count || inputs.empty() ||
      inputs.size() > VirtualPipelineState::InputCapacity) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (!reset_device_vsm_run_evidence(*owner)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  run.owner = owner;
  run.state = &state;
  run.input_count = inputs.size();
  run.output = &output;
  run.projection = &projection;
  run.stats = &stats;
  run.registration.reset();
  run.lease.reset();
  run.snapshot = {};
  run.submission_control.reset();
  run.request = {};
  run.pipeline_started = {};
  run.failed_input = ResidencyStats::no_failed_page;
  run.failed_page = ResidencyStats::no_failed_page;
  run.backing_recovery = false;
  run.backing_may_write = false;
  run.publication_success = false;
  run.output_hash_observed = false;
  run.output_hash = 0u;
  run.callback_count.store(0u, std::memory_order_release);
  run.done.store(false, std::memory_order_release);
  std::copy(inputs.begin(), inputs.end(), run.inputs.begin());
  for (std::size_t index = 0u; index < run.input_count; ++index) {
    std::vector<std::byte> &staged = owner->input_staging[index];
    run.input_staging[index] =
        std::span<std::byte>{staged.data(), staged.size()};
  }
  run.output_staging = std::span<std::byte>{owner->output_staging.data(),
                                            owner->output_staging.size()};
  const Status snapshot =
      snapshot_pipelines({owner->pipelines.data(), owner->pipeline_count},
                         {owner->pipeline_stages.data(), owner->pipeline_count},
                         projection.graph_execution, run.snapshot);
  if (!snapshot) {
    owner->submitted = false;
    if (snapshot.reason() == Reason::DeviceLost) {
      quarantine_owner(owner);
      run.poison_pipeline = true;
    }
    return snapshot;
  }
  const Status staged = stage_input(run);
  if (!staged) {
    run.poison_pipeline = staged.reason() == Reason::DeviceLost;
    if (run.poison_pipeline) {
      quarantine_owner(owner);
    }
    return staged;
  }
  const Status pipelines = begin_pipelines(run);
  if (!pipelines) {
    run.poison_pipeline = pipelines.reason() == Reason::DeviceLost;
    if (run.poison_pipeline) {
      quarantine_owner(owner);
    }
    return pipelines;
  }
  run.registration = owner->registration;
  if (run.registration == nullptr) {
    const Status failure = Status::fail(Reason::PipelineMemoryBudget);
    reject_pipelines(run, failure);
    return failure;
  }
  run.lease.emplace(
      state.pipeline->device->residency->authority().direct_recurrences().begin_direct_recurrence(
          run.registration->request()));
  if (!run.lease.has_value() || !*run.lease) {
    const Status failure = Status::fail(Reason::PipelineBusy);
    reject_pipelines(run, failure);
    return failure;
  }
  VirtualBackingAccess::require_recovery(output,
                                         projection.active.output_bytes);
  run.backing_recovery = true;
  return submit(run);
}

namespace {

[[nodiscard]] bool open(
    const device_vsm_product_detail::DeviceVsmProductRun &run) noexcept {
  return device_vsm_product_detail::lifecycle_open(run);
}

[[nodiscard]] bool valid(
    const device_vsm_product_detail::DeviceVsmProductRun &run) noexcept {
  return run.done.load(std::memory_order_acquire) &&
         run.callback_count.load(std::memory_order_acquire) == 1u &&
         run.submission_control.count() == 1u &&
         node::accel::detail::device_vsm_final_valid(run.request, run.final);
}

void unknown(device_vsm_product_detail::DeviceVsmProductRun &run) noexcept {
  run.final = device_vsm_product_detail::unknown_final(run);
  run.poison_pipeline = true;
}

const VirtualVsmAsyncOps AsyncOps{
    .start = start_accel_virtual_execution_device_vsm,
    .finish = device_vsm_product_detail::finish,
    .open = open,
    .valid = valid,
    .unknown = unknown,
};

} // namespace

const VirtualVsmAsyncOps &accel_virtual_vsm_async_ops() noexcept {
  return AsyncOps;
}

VirtualExecutionResult execute_accel_virtual_execution_device_vsm(
    VirtualPipelineState &state, const std::span<VirtualBacking *const> inputs,
    VirtualBacking &output, const VirtualRunProjection &projection,
    const VirtualExecutionDeviceVsmPrepared &prepared, Stats &stats) noexcept {
  device_vsm_product_detail::DeviceVsmProductRun run{.stats = &stats};
  const Status started = start_accel_virtual_execution_device_vsm(
      state, inputs, output, projection, prepared, stats, run);
  if (!started && run.submission_control.count() == 0u &&
      !lifecycle_open(run)) {
    return VirtualExecutionResult{
        .status = started,
        .failed_page = run.failed_page,
        .poison_pipeline = run.poison_pipeline,
        .certainty = VirtualRunWriteCertainty::KnownNoWrite,
    };
  }
  return finish(run, started);
}

} // namespace rund::compute::detail
