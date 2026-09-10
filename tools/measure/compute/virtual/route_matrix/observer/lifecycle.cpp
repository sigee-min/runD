#include "local.hpp"

#include <memory>

namespace rund::measure::compute::route_matrix {

RouteObserver::RouteObserver(::rund::compute::Device &device) noexcept {
  try {
    impl_ = std::make_unique<RouteObserverImpl>();
  } catch (...) {
    return;
  }
  const auto &state = ::rund::compute::detail::DeviceAccess::state(device);
  if (state == nullptr || state->ops == nullptr ||
      active_route_observer != nullptr) {
    return;
  }
  impl_->device = state.get();
  impl_->original = state->ops;
  const bool prepare =
      impl_->original->virtual_execution.prepare_virtual_device_vsm_product !=
      nullptr;
  const bool execute =
      impl_->original->virtual_execution.execute_virtual_device_vsm_product !=
      nullptr;
  impl_->evidence.production_slots = prepare && execute;
  if (!impl_->evidence.production_slots) {
    return;
  }
  impl_->routed = *impl_->original;
  impl_->routed.virtual_execution.prepare_virtual_device_vsm_product =
      prepare_route_observed;
  impl_->routed.virtual_execution.execute_virtual_device_vsm_product =
      execute_route_observed;
  impl_->device->ops = &impl_->routed;
  active_route_observer = impl_.get();
  impl_->active = true;
}

RouteObserver::~RouteObserver() {
  if (impl_ == nullptr || !impl_->active) {
    return;
  }
  if (impl_->device != nullptr) {
    impl_->device->ops = impl_->original;
  }
  if (active_route_observer == impl_.get()) {
    active_route_observer = nullptr;
  }
}

void RouteObserver::reset() noexcept {
  if (impl_ == nullptr) {
    return;
  }
  const bool production_slots = impl_->evidence.production_slots;
  impl_->evidence = {};
  impl_->evidence.production_slots = production_slots;
  impl_->owner_seen = false;
  impl_->owner_token = 0u;
  impl_->control_token = 0u;
  impl_->previous_hash_observations = 0u;
  impl_->previous_hash_reuses = 0u;
  impl_->proof_flags = 0u;
  impl_->capability_flags = 0u;
  impl_->capability_width = 0u;
  impl_->capability_retained = 0u;
  impl_->capability_transient = 0u;
  impl_->capability_seen = false;
  impl_->proof_identity_seen = false;
  impl_->prepared_count = 0u;
  impl_->executed_count = 0u;
  impl_->pending_prepared = false;
  impl_->lifecycle_seen = false;
  impl_->lifecycle_sealed = false;
}

void RouteObserver::finish() noexcept {
  if (impl_ == nullptr || impl_->lifecycle_sealed) {
    return;
  }
  impl_->lifecycle_sealed = true;
  const bool complete = !impl_->pending_prepared &&
                        impl_->prepared_count == CohortRuns &&
                        impl_->executed_count == CohortRuns &&
                        impl_->evidence.lifecycle_observations == CohortRuns &&
                        impl_->evidence.lifecycle_mismatches == 0u;
  if (!complete) {
    mark_lifecycle_failure(*impl_);
  }
  impl_->evidence.lifecycle_complete = complete;
}

const RouteEvidence &RouteObserver::evidence() const noexcept {
  static const RouteEvidence unavailable{};
  return impl_ == nullptr ? unavailable : impl_->evidence;
}

} // namespace rund::measure::compute::route_matrix
