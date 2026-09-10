#include "local.hpp"

#include "src/compute/virtual/run/device_vsm/model.hpp"

#include <algorithm>
#include <limits>
#include <memory>

namespace rund::measure::compute::route_matrix {

thread_local RouteObserverImpl *active_route_observer{};

namespace {

void add(std::uint64_t &target, const std::uint64_t value) noexcept {
  target = value > std::numeric_limits<std::uint64_t>::max() - target
               ? std::numeric_limits<std::uint64_t>::max()
               : target + value;
}

[[nodiscard]] std::uint32_t proof_bits(
    const ::rund::compute::detail::VirtualDeviceVsmRouteProof &proof) noexcept {
  const bool staged_loop =
      proof.kind ==
      ::rund::compute::detail::VirtualDeviceVsmRouteKind::StagedLoop;
  const bool graph_resident =
      proof.kind ==
      ::rund::compute::detail::VirtualDeviceVsmRouteKind::GraphResident;
  const std::uint32_t mode =
      proof.kind ==
              ::rund::compute::detail::VirtualDeviceVsmRouteKind::WindowRing
          ? static_cast<std::uint32_t>(WindowRingMode)
          : 0u;
  return (staged_loop ? 1u : 0u) | (graph_resident ? 2u : 0u) | (mode << 2u) |
         (static_cast<std::uint32_t>(proof.endpoint) << 4u);
}

[[nodiscard]] std::uint32_t capability_bits(const auto &capability) noexcept {
  return (capability.device_generated_recurrence ? 1u : 0u) |
         (capability.fixed_native_storage ? 2u : 0u) |
         (capability.fixed_common_storage ? 4u : 0u) |
         (capability.gpu_addressable_backing ? 8u : 0u) |
         (capability.one_native_submit ? 16u : 0u) |
         (capability.host_service_turns_zero ? 32u : 0u) |
         (capability.host_epoch_callbacks_zero ? 64u : 0u) |
         (capability.aggregate_terminal_once ? 128u : 0u) |
         (capability.bounded_page_io ? 256u : 0u) |
         (capability.physical_ring_storage ? 512u : 0u);
}

} // namespace

void observe_proof(
    RouteObserverImpl &observer,
    const ::rund::compute::detail::VirtualDeviceVsmRouteProof &proof) noexcept {
  auto &evidence = observer.evidence;
  const bool valid = proof.valid();
  const std::uint32_t flags = proof_bits(proof);
  add(evidence.proof_valid_count, valid ? 1u : 0u);
  if (!evidence.proof_seen) {
    evidence.proof_seen = true;
    evidence.proof_valid = valid;
    evidence.proof_staged_loop =
        proof.kind ==
        ::rund::compute::detail::VirtualDeviceVsmRouteKind::StagedLoop;
    evidence.proof_graph_resident =
        proof.kind ==
        ::rund::compute::detail::VirtualDeviceVsmRouteKind::GraphResident;
    evidence.proof_mode =
        proof.kind ==
                ::rund::compute::detail::VirtualDeviceVsmRouteKind::WindowRing
            ? WindowRingMode
            : 0u;
    evidence.proof_endpoint = static_cast<std::uint64_t>(proof.endpoint);
    evidence.proof_identity_stable = true;
    evidence.proof_flags_stable = true;
    observer.proof_flags = flags;
    return;
  }
  evidence.proof_valid = evidence.proof_valid && valid;
  if (observer.proof_flags != flags) {
    evidence.proof_flags_stable = false;
    add(evidence.proof_flag_mismatches, 1u);
  }
}

namespace {

void observe_identity(RouteObserverImpl &observer, const std::uint64_t hi,
                      const std::uint64_t lo) noexcept {
  auto &evidence = observer.evidence;
  if (!observer.proof_identity_seen) {
    observer.proof_identity_seen = true;
    evidence.proof_hi = hi;
    evidence.proof_lo = lo;
    evidence.proof_identity_stable = true;
    return;
  }
  if (evidence.proof_hi != hi || evidence.proof_lo != lo) {
    evidence.proof_identity_stable = false;
    add(evidence.proof_identity_mismatches, 1u);
  }
}

void observe_capability(RouteObserverImpl &observer,
                        const auto &capability) noexcept {
  auto &evidence = observer.evidence;
  const std::uint32_t flags = capability_bits(capability);
  add(evidence.capability_observations, 1u);
  if (!observer.capability_seen) {
    observer.capability_seen = true;
    observer.capability_flags = flags;
    observer.capability_width = capability.width;
    observer.capability_retained = capability.retained_bytes;
    observer.capability_transient = capability.transient_bytes;
    evidence.capability_consistent = true;
    return;
  }
  if (observer.capability_flags != flags ||
      observer.capability_width != capability.width ||
      observer.capability_retained != capability.retained_bytes ||
      observer.capability_transient != capability.transient_bytes) {
    evidence.capability_consistent = false;
    add(evidence.capability_mismatches, 1u);
  }
}

void mark_lifecycle(RouteObserverImpl &observer, const bool valid) noexcept {
  auto &evidence = observer.evidence;
  add(evidence.lifecycle_observations, 1u);
  if (!observer.lifecycle_seen) {
    observer.lifecycle_seen = true;
    evidence.per_run_lifecycle_consistent = true;
  }
  if (!valid) {
    evidence.per_run_lifecycle_consistent = false;
    add(evidence.lifecycle_mismatches, 1u);
  }
}

void observe_hash(RouteObserverImpl &observer, const auto &product,
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

void mark_lifecycle_failure(RouteObserverImpl &observer) noexcept {
  auto &evidence = observer.evidence;
  if (!observer.lifecycle_seen) {
    observer.lifecycle_seen = true;
    evidence.per_run_lifecycle_consistent = true;
  }
  evidence.per_run_lifecycle_consistent = false;
  add(evidence.lifecycle_mismatches, 1u);
}

void RouteObserverImpl::observe_owner(
    const ::rund::compute::detail::VirtualExecutionDeviceVsmPrepared
        &prepared) noexcept {
  using Owner =
      ::rund::compute::detail::device_vsm_product_detail::DeviceVsmProductOwner;
  if (evidence.owner_events == 0u) {
    evidence.owner_stable = true;
    previous_hash_observations = 0u;
    previous_hash_reuses = 0u;
  }
  ++evidence.owner_events;
  const auto owner = std::static_pointer_cast<Owner>(prepared.owner);
  const std::uintptr_t owner_token =
      reinterpret_cast<std::uintptr_t>(owner.get());
  const std::uintptr_t control_token =
      owner == nullptr
          ? 0u
          : reinterpret_cast<std::uintptr_t>(owner->registration.get());
  if (owner != nullptr) {
    if (owner->proof != nullptr) {
      observe_identity(*this, owner->proof->identity.hi,
                       owner->proof->identity.lo);
    }
    observe_capability(*this, owner->preparation.capability);
  }
  if (!owner_seen) {
    owner_seen = true;
    this->owner_token = owner_token;
    this->control_token = control_token;
  } else if (this->owner_token != owner_token ||
             this->control_token != control_token) {
    evidence.owner_stable = false;
  }
}

void RouteObserverImpl::observe(
    const ::rund::compute::detail::VirtualExecutionDeviceVsmPrepared &prepared,
    const ::rund::compute::detail::VirtualExecutionResult &result) noexcept {
  using Owner =
      ::rund::compute::detail::device_vsm_product_detail::DeviceVsmProductOwner;
  const bool paired = pending_prepared;
  pending_prepared = false;
  if (executed_count >= CohortRuns) {
    mark_lifecycle_failure(*this);
  } else {
    ++executed_count;
  }
  ++evidence.executed_runs;
  const auto owner = std::static_pointer_cast<Owner>(prepared.owner);
  bool lifecycle = paired && owner != nullptr && owner->evidence != nullptr;
  if (lifecycle) {
    const auto &product = *owner->evidence;
    lifecycle = result.status && product.final_received &&
                !product.quarantined && product.public_handoff_count == 1u &&
                product.authority_accept_count == 1u &&
                product.pipeline_terminal_count == owner->pipeline_count &&
                product.native.native_submit_count == 1u &&
                product.native.epoch_native_submit_count == 0u &&
                product.native.host_service_turn_count == 0u &&
                product.native.host_epoch_callback_count == 0u &&
                product.native.final_callback_count == 1u &&
                owner->preparation.capability.aggregate_terminal_once &&
                product.backing_publication_count == 1u;
  }
  mark_lifecycle(*this, lifecycle);
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
  evidence.physical_ring_storage_runs +=
      capability.physical_ring_storage ? 1u : 0u;
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
  add(evidence.coordinate_count, native.page_count);
  add(evidence.accepted_coordinates, native.generated_epochs);
  add(evidence.gpu_completed_coordinates, native.completed_epochs);
  add(evidence.completed_prefix, native.completed_epochs);
  add(evidence.forecasted_pages, native.forecasted_pages);
  add(evidence.promoted_pages, native.promoted_pages);
  add(evidence.drained_pages, native.drained_pages);
  add(evidence.persisted_pages, native.persisted_pages);
  add(evidence.overlap_reused_bytes, native.overlap_reused_bytes);
  add(evidence.gpu_backing_read_bytes, native.gpu_backing_read_bytes);
  add(evidence.gpu_backing_write_bytes, native.gpu_backing_write_bytes);
  add(evidence.window_seed_dispatches, native.window_seed_dispatches);
  add(evidence.window_compute_dispatches, native.window_compute_dispatches);
  add(evidence.window_internal_dispatches, native.window_internal_dispatches);
  add(evidence.native_submit_count, native.native_submit_count);
  add(evidence.epoch_native_submit_count, native.epoch_native_submit_count);
  add(evidence.payload_dispatch_count, native.payload_dispatch_count);
  add(evidence.host_service_turn_count, native.host_service_turn_count);
  add(evidence.host_epoch_callback_count, native.host_epoch_callback_count);
  add(evidence.final_callback_count, native.final_callback_count);
  add(evidence.queue_calls, native.native_submit_count);
  add(evidence.public_handoff_count, product.public_handoff_count);
  add(evidence.authority_accept_count, product.authority_accept_count);
  add(evidence.pipeline_terminal_count, product.pipeline_terminal_count);
  add(evidence.backing_publication_count, product.backing_publication_count);
  add(evidence.completed_ns, native.completed_ns);
  evidence.retained_bytes =
      std::max(evidence.retained_bytes, capability.retained_bytes);
  evidence.transient_bytes =
      std::max(evidence.transient_bytes, capability.transient_bytes);
}

} // namespace rund::measure::compute::route_matrix
