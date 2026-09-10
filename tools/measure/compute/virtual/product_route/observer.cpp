#include "internal.hpp"

#include "src/compute/virtual/run/device_vsm.hpp"

namespace rund::measure::compute::virtual_residency {
using ::rund::compute::Stats;
using ::rund::compute::Status;
using ::rund::compute::VirtualBacking;
using ::rund::compute::detail::VirtualDeviceVsmRouteProof;
using ::rund::compute::detail::VirtualExecutionDeviceVsmPrepared;
using ::rund::compute::detail::VirtualExecutionResult;
using ::rund::compute::detail::VirtualPipelineState;
using ::rund::compute::detail::VirtualRunProjection;

namespace {

void observe_hash(ProductRouteObserverImpl &observer, const auto &product,
                  const bool resident_output) noexcept {
  auto &evidence = observer.evidence;
  const std::uint64_t observations = product.output_hash_observation_count;
  const std::uint64_t reuses = product.output_hash_reuse_count;
  const bool first = evidence.executed_runs == 1u;
  if (first) {
    evidence.first_output_hash_observation_count = observations;
    evidence.first_output_hash_reuse_count = reuses;
    observer.previous_hash_observations = observations;
    observer.previous_hash_reuses = reuses;
    if (!resident_output) {
      if (observations != 0u) {
        ++evidence.hash_observation_changes;
      }
      if (reuses != 0u) {
        ++evidence.hash_reuse_step_errors;
      }
    }
  } else if (resident_output) {
    if (observations != observer.previous_hash_observations) {
      ++evidence.hash_observation_changes;
    }
    if (observer.previous_hash_reuses ==
            std::numeric_limits<std::uint64_t>::max() ||
        reuses != observer.previous_hash_reuses + 1u) {
      ++evidence.hash_reuse_step_errors;
    }
    observer.previous_hash_observations = observations;
    observer.previous_hash_reuses = reuses;
  } else {
    if (observations != 0u) {
      ++evidence.hash_observation_changes;
    }
    if (reuses != 0u) {
      ++evidence.hash_reuse_step_errors;
    }
    observer.previous_hash_observations = observations;
    observer.previous_hash_reuses = reuses;
  }
  evidence.last_output_hash_observation_count = observations;
  evidence.last_output_hash_reuse_count = reuses;
}

} // namespace

Status prepare_product_route_observed(
    VirtualPipelineState &state, const VirtualRunProjection &run,
    const VirtualDeviceVsmRouteProof proof,
    VirtualExecutionDeviceVsmPrepared &prepared) noexcept {
  ProductRouteObserverImpl *const observer = active_product_route_observer;
  if (observer == nullptr || observer->original == nullptr ||
      observer->original->virtual_execution
              .prepare_virtual_device_vsm_product == nullptr) {
    return Status::fail(::rund::compute::Reason::BackendUnsupported);
  }
  ++observer->evidence.prepare_attempts;
  const Status status =
      observer->original->virtual_execution.prepare_virtual_device_vsm_product(
          state, run, proof, prepared);
  observer->evidence.backend = observer->device == nullptr
                                   ? ::rund::compute::Backend::Unavailable
                                   : observer->device->backend;
  observer->observe_owner(prepared);
  if (status) {
    ++observer->evidence.prepared_runs;
  }
  return status;
}

VirtualExecutionResult execute_product_route_observed(
    VirtualPipelineState &state, const std::span<VirtualBacking *const> inputs,
    VirtualBacking &output, const VirtualRunProjection &run,
    const VirtualExecutionDeviceVsmPrepared &prepared, Stats &stats) noexcept {
  ProductRouteObserverImpl *const observer = active_product_route_observer;
  if (observer == nullptr || observer->original == nullptr ||
      observer->original->virtual_execution
              .execute_virtual_device_vsm_product == nullptr) {
    return VirtualExecutionResult{
        .status = Status::fail(::rund::compute::Reason::BackendUnsupported)};
  }
  observer->evidence.backend = observer->device == nullptr
                                   ? ::rund::compute::Backend::Unavailable
                                   : observer->device->backend;
  observer->observe_owner(prepared);
  const VirtualExecutionResult result =
      observer->original->virtual_execution.execute_virtual_device_vsm_product(
          state, inputs, output, run, prepared, stats);
  observer->observe(prepared, result);
  return result;
}

void ProductRouteObserverImpl::observe_owner(
    const VirtualExecutionDeviceVsmPrepared &prepared) noexcept {
  using ::rund::compute::detail::device_vsm_product_detail::
      DeviceVsmProductOwner;
  if (evidence.owner_events == 0u) {
    evidence.owner_stable = true;
    previous_hash_observations = 0u;
    previous_hash_reuses = 0u;
  }
  ++evidence.owner_events;
  const auto owner =
      std::static_pointer_cast<DeviceVsmProductOwner>(prepared.owner);
  const std::uintptr_t owner_token =
      reinterpret_cast<std::uintptr_t>(owner.get());
  const std::uintptr_t control_token =
      owner == nullptr
          ? 0u
          : reinterpret_cast<std::uintptr_t>(owner->registration.get());
  if (!owner_seen) {
    owner_seen = true;
    this->owner_token = owner_token;
    this->control_token = control_token;
  } else if (this->owner_token != owner_token ||
             this->control_token != control_token) {
    evidence.owner_stable = false;
  }
}

void ProductRouteObserverImpl::observe(
    const ::rund::compute::detail::VirtualExecutionDeviceVsmPrepared &prepared,
    const ::rund::compute::detail::VirtualExecutionResult &result) noexcept {
  using ::rund::compute::detail::device_vsm_product_detail::
      DeviceVsmProductOwner;
  ++evidence.executed_runs;
  const auto owner =
      std::static_pointer_cast<DeviceVsmProductOwner>(prepared.owner);
  if (owner == nullptr || owner->evidence == nullptr ||
      !owner->evidence->final_received) {
    return;
  }
  const auto &product = *owner->evidence;
  const auto &capability = owner->preparation.capability;
  const auto &native = product.native;
  observe_hash(*this, product, product.public_resident_output);
  if (result.status && native.may_write && !product.quarantined) {
    ++evidence.successful_finals;
  }
  evidence.cold_owner_runs += product.warm_rearm_count == 0u ? 1u : 0u;
  evidence.warm_reused_runs += product.warm_rearm_count != 0u ? 1u : 0u;
  evidence.max_warm_rearm_count =
      std::max(evidence.max_warm_rearm_count, product.warm_rearm_count);
  evidence.device_generated_recurrence_runs +=
      capability.device_generated_recurrence ? 1u : 0u;
  evidence.fixed_native_storage_runs +=
      capability.fixed_native_storage ? 1u : 0u;
  evidence.fixed_common_storage_runs +=
      capability.fixed_common_storage ? 1u : 0u;
  evidence.gpu_addressable_backing_runs +=
      capability.gpu_addressable_backing ? 1u : 0u;
  const bool public_resident =
      product.public_resident_input_count == owner->input_count &&
      product.public_resident_output;
  const bool whole_run_staged = product.whole_run_staged_input_count != 0u ||
                                product.whole_run_staged_output;
  evidence.public_gpu_addressable_backing_runs += public_resident ? 1u : 0u;
  evidence.whole_run_staging_runs += whole_run_staged ? 1u : 0u;
  evidence.bounded_external_page_service_runs +=
      product.bounded_external_page_service ? 1u : 0u;
  evidence.one_native_submit_runs += capability.one_native_submit ? 1u : 0u;
  evidence.host_service_turns_zero_runs +=
      capability.host_service_turns_zero ? 1u : 0u;
  evidence.host_epoch_callbacks_zero_runs +=
      capability.host_epoch_callbacks_zero ? 1u : 0u;
  evidence.aggregate_terminal_once_runs +=
      capability.aggregate_terminal_once ? 1u : 0u;
  evidence.bounded_page_io_runs += capability.bounded_page_io ? 1u : 0u;
  add_saturated(evidence.coordinate_count, native.page_count);
  add_saturated(evidence.accepted_coordinates, native.generated_epochs);
  add_saturated(evidence.gpu_completed_coordinates, native.completed_epochs);
  add_saturated(evidence.completed_prefix, native.completed_epochs);
  add_saturated(evidence.forecasted_pages, native.forecasted_pages);
  add_saturated(evidence.promoted_pages, native.promoted_pages);
  add_saturated(evidence.drained_pages, native.drained_pages);
  add_saturated(evidence.persisted_pages, native.persisted_pages);
  add_saturated(evidence.overlap_reused_bytes, native.overlap_reused_bytes);
  add_saturated(evidence.gpu_backing_read_bytes, native.gpu_backing_read_bytes);
  add_saturated(evidence.gpu_backing_write_bytes,
                native.gpu_backing_write_bytes);
  add_saturated(evidence.native_submit_count, native.native_submit_count);
  add_saturated(evidence.epoch_native_submit_count,
                native.epoch_native_submit_count);
  add_saturated(evidence.payload_dispatch_count, native.payload_dispatch_count);
  add_saturated(evidence.host_service_turn_count,
                native.host_service_turn_count);
  add_saturated(evidence.host_epoch_callback_count,
                native.host_epoch_callback_count);
  add_saturated(evidence.final_callback_count, native.final_callback_count);
  add_saturated(evidence.queue_calls, native.native_submit_count);
  add_saturated(evidence.public_handoff_count, product.public_handoff_count);
  add_saturated(evidence.authority_accept_count,
                product.authority_accept_count);
  add_saturated(evidence.pipeline_terminal_count,
                product.pipeline_terminal_count);
  add_saturated(evidence.backing_publication_count,
                product.backing_publication_count);
  add_saturated(evidence.completed_ns, native.completed_ns);
  evidence.retained_bytes =
      std::max(evidence.retained_bytes, capability.retained_bytes);
  evidence.transient_bytes =
      std::max(evidence.transient_bytes, capability.transient_bytes);
}

} // namespace rund::measure::compute::virtual_residency
